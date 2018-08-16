from __future__ import division
import tensorflow as tf
from general import get_initializer
from tf_utils import create_calc_similarity_fn, F, gather
use_cudnn_rnn = True 
if use_cudnn_rnn:
    from tf_utils import cudnn_rnn as rnn
else:
    from tf_utils import rnn
from nn import softsel, multi_conv1d, linear, get_logits, highway_network
from base import ModelBase

class Model(ModelBase):
    def setupEmbedding(self):
        def emb_drop(E, x):
            emb = tf.nn.embedding_lookup(E, x)
            emb_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(emb, self.cfg.input_keep_prob), lambda: emb)
            return emb_drop
        
        cfg = self.cfg
        VW, dw, d = cfg.vocab_size, cfg.embedding_size, cfg.hidden_size
        VC, W, dc, dco = cfg.char_vocab_size, cfg.max_word_len, cfg.char_emb_dim, cfg.char_out_size
        N, JX, JQ = tf.shape(self.premise_x)[0], tf.shape(self.premise_x)[1], tf.shape(self.hypothesis_x)[1]
        with tf.variable_scope('emb'):
            with tf.variable_scope('emb_var'), tf.device('/cpu:0'):
                word_emb_mat = tf.get_variable('word_emb_mat', shape=[VW, dw], dtype='float', 
                                                initializer=get_initializer(cfg.embeddings), trainable=cfg.tune_embedding)
                char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float', 
                                                    initializer=get_initializer(cfg.char_embeddings), trainable=True)

            with tf.variable_scope('word'):
                # premise_in = emb_drop(word_emb_mat, self.premise_x)   #P
                # hypothesis_in = emb_drop(word_emb_mat, self.hypothesis_x)  #H
                premise_in = tf.nn.embedding_lookup(word_emb_mat, self.premise_x)
                hypothesis_in = tf.nn.embedding_lookup(word_emb_mat, self.hypothesis_x)


            with tf.variable_scope("char"):
                # char_pre = emb_drop(char_emb_mat, self.premise_char)
                # char_hyp = emb_drop(char_emb_mat, self.hypothesis_char)
                char_pre = tf.nn.embedding_lookup(char_emb_mat, self.premise_char)
                char_hyp = tf.nn.embedding_lookup(char_emb_mat, self.hypothesis_char)

                filter_sizes = list(map(int, cfg.out_channel_dims.split(','))) #[100]
                heights = list(map(int, cfg.filter_heights.split(',')))        #[5]
                assert sum(filter_sizes) == cfg.char_out_size, (filter_sizes, cfg.char_out_size)
                with tf.variable_scope("conv") as scope:
                    conv_pre = multi_conv1d(char_pre, filter_sizes, heights, "VALID", 
                                            is_train=self.is_train, 
                                            keep_prob=cfg.input_keep_prob, 
                                            wd=cfg.wd, 
                                            scope='conv')
                    scope.reuse_variables()  
                    conv_hyp = multi_conv1d(char_hyp, filter_sizes, heights, "VALID", 
                                            is_train=self.is_train, 
                                            keep_prob=cfg.input_keep_prob, 
                                            scope='conv')
                    conv_pre = tf.reshape(conv_pre, [-1, JX, cfg.char_out_size])
                    conv_hyp = tf.reshape(conv_hyp, [-1, JQ, cfg.char_out_size])

            premise_in = tf.concat([premise_in, conv_pre], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, conv_hyp], axis=2)

            if cfg.use_additional_features:
                premise_in = tf.concat((premise_in, tf.cast(self.premise_pos, tf.float32)), axis=2)
                hypothesis_in = tf.concat((hypothesis_in, tf.cast(self.hypothesis_pos, tf.float32)), axis=2)

                premise_in = tf.concat([premise_in, tf.cast(self.premise_exact_match, tf.float32)], axis=2)
                hypothesis_in = tf.concat([hypothesis_in, tf.cast(self.hypothesis_exact_match, tf.float32)], axis=2)
            
            dw = premise_in.get_shape().as_list()[-1]
            
            return premise_in, hypothesis_in, dw

    def buildModel(self):
        
        cfg = self.cfg
        d = cfg.hidden_size
        N, JX, JQ = tf.shape(self.premise_x)[0], tf.shape(self.premise_x)[1], tf.shape(self.hypothesis_x)[1]
        premise_in, hypothesis_in, dw = self.setupEmbedding()
        premise_len = tf.reduce_sum(tf.cast(self.premise_mask, tf.int32), 1)
        hypothesis_len = tf.reduce_sum(tf.cast(self.hypothesis_mask, tf.int32), 1)
        calc_similarity_fn = create_calc_similarity_fn(cfg, self.is_train)
        
        # import ipdb; ipdb.set_trace()

        with tf.variable_scope("projection"):
            premise_in = F(premise_in, 2*d, activation=tf.nn.relu, scope='proj_premise_in', input_keep_prob=cfg.input_keep_prob, is_train=self.is_train, wd=cfg.wd)
            tf.get_variable_scope().reuse_variables()
            hypothesis_in = F(hypothesis_in, 2*d, activation=tf.nn.relu, scope='proj_premise_in', input_keep_prob=cfg.input_keep_prob, is_train=self.is_train)
        
        with tf.variable_scope("encoding"):
            ph, _ = rnn(cfg.rnn_type, tf.concat([premise_in, hypothesis_in], 0), premise_len, d, scope='premise_in_rnn', num_layers=cfg.num_rnn_layers, 
                        dropout_keep_prob=cfg.input_keep_prob, is_train=self.is_train, wd=cfg.wd) # [N x JQ x 2d]
            rnn_premise_in, rnn_hypothesis_in = tf.split(ph, 2)
            # rnn_premise_in, _ = rnn(cfg.rnn_type, premise_in, premise_len, d, scope='premise_in_rnn', num_layers=cfg.num_rnn_layers, 
            #                         dropout_keep_prob=cfg.input_keep_prob, is_train=self.is_train, wd=cfg.wd) # [N x JQ x 2d]
            # tf.get_variable_scope().reuse_variables()
            # rnn_hypothesis_in, _ = rnn(cfg.rnn_type, hypothesis_in, hypothesis_len, d, scope='premise_in_rnn', num_layers=cfg.num_rnn_layers, 
            #                         dropout_keep_prob=cfg.input_keep_prob, is_train=self.is_train) # [N x JQ x 2d]
            
            rnn_premise_in = rnn_premise_in * tf.to_float(tf.expand_dims(self.premise_mask, -1))
            rnn_hypothesis_in = rnn_hypothesis_in * tf.to_float(tf.expand_dims(self.hypothesis_mask, -1))

        with tf.variable_scope('fuse_gate'):
            premise_in = fuse_gate_v2(premise_in, rnn_premise_in, scope="premise_in_fuse_gate", is_train=self.is_train, input_keep_prob=cfg.input_keep_prob, wd=cfg.wd)
            tf.get_variable_scope().reuse_variables()
            hypothesis_in = fuse_gate_v2(hypothesis_in, rnn_hypothesis_in, scope="premise_in_fuse_gate", is_train=self.is_train, input_keep_prob=cfg.input_keep_prob)

        with tf.variable_scope("main"):
            d = premise_in.get_shape().as_list()[-1]
            h_max_list = []
            p_max_list = []
            p_mask = self.premise_mask
            h_mask = self.hypothesis_mask
            p, h = premise_in, hypothesis_in
            for i in range(cfg.num_layers):
                p, h, p_max, h_max = basic_block(cfg, p, h, p_mask, h_mask, 
                                                input_keep_prob=cfg.input_keep_prob, 
                                                is_train=self.is_train, 
                                                scope="basic_block_{}".format(i))
                p_max_list.append(p_max)
                h_max_list.append(h_max)

        self_attention = True
        if self_attention:
            p_maxs = tf.stack(p_max_list, 1)
            h_maxs = tf.stack(h_max_list, 1)

            p_maxs_len = len(p_max_list) * tf.ones_like(p_max[:,0], dtype=tf.int32)
            with tf.variable_scope("rnn_aggregate"):
                ph, _ = rnn(cfg.rnn_type, tf.concat([p_maxs, h_maxs], 0), p_maxs_len, d, scope='p_maxs', 
                            dropout_keep_prob=cfg.input_keep_prob, is_train=self.is_train, wd=cfg.wd)
                p_maxs, h_maxs = tf.split(ph, 2)
                
                # p_maxs, _ = rnn(cfg.rnn_type, p_maxs, p_maxs_len, d, scope='p_maxs', 
                #                 dropout_keep_prob=cfg.input_keep_prob, is_train=self.is_train, wd=cfg.wd)
                # tf.get_variable_scope().reuse_variables()
                # h_maxs, _ = rnn(cfg.rnn_type, h_maxs, p_maxs_len, d, scope='p_maxs', 
                #                 dropout_keep_prob=cfg.input_keep_prob, is_train=self.is_train)
            
            _, _, p_max2, h_max2 = basic_block(cfg, p_maxs, h_maxs, 
                                            tf.ones_like(p_maxs[:,:,0], dtype=tf.bool), 
                                            tf.ones_like(h_maxs[:,:,0], dtype=tf.bool), 
                                            is_train=self.is_train, 
                                            scope='basic_block_att')
            p_max = tf.concat([p_max, p_max2], -1)
            h_max = tf.concat([h_max, h_max2], -1)

        f0 = tf.concat([p_max, h_max, p_max * h_max, tf.abs(p_max - h_max)], -1)
        d = p_max.get_shape().as_list()[-1]
        f0 = F(f0, d, activation=tf.nn.tanh, scope='f0', 
            input_keep_prob=cfg.input_keep_prob, is_train=self.is_train, wd=cfg.wd)
        
        self.logits = linear(f0, cfg.pred_size, True, bias_start=0.0, scope="logit", squeeze=False, 
                            wd=cfg.wd, input_keep_prob=cfg.input_keep_prob, is_train=self.is_train)
        self.yp = tf.argmax(self.logits, 1)
        self.accuracy = tf.reduce_sum(tf.to_int32(tf.equal(tf.argmax(self.y, 1), self.yp)))


def self_attention(config, is_train, p, p_mask, scope=None): #[N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):
        PL = tf.shape(p)[1]
        d = p.get_shape().as_list()[-1]
        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1, 1, PL, 1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1, PL, 1, 1]) #[N, PL, HL, 2d]

        p_mask_aug_1 = tf.tile(tf.expand_dims(p_mask, 2), [1, 1, PL])
        p_mask_aug_2 = tf.tile(tf.expand_dims(p_mask, 1), [1, PL, 1])
        self_mask = p_mask_aug_1 & p_mask_aug_2

        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                              is_train=is_train, func=config.att_func, scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits, mask=self_mask, scope='self_att') 

        return self_att

def fuse_gate(config, is_train, lhs, rhs, scope=None):
    with tf.variable_scope(scope or "fuse_gate"):
        dim = lhs.get_shape().as_list()[-1]
        lhs_1 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_1", squeeze=False, wd=config.wd, input_keep_prob=config.input_keep_prob, is_train=is_train)
        rhs_1 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_1", squeeze=False, wd=0.0, input_keep_prob=config.input_keep_prob, is_train=is_train)
        z = tf.tanh(lhs_1 + rhs_1)
        lhs_2 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_2", squeeze=False, wd=config.wd, input_keep_prob=config.input_keep_prob, is_train=is_train)
        rhs_2 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_2", squeeze=False, wd=config.wd, input_keep_prob=config.input_keep_prob, is_train=is_train)
        f = tf.sigmoid(lhs_2 + rhs_2)
        out = f * lhs + (1 - f) * z
        return out

def fuse_gate_v2(lhs, rhs, is_train=None, wd=0.0, input_keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "fuse_gate"):
        dim = lhs.get_shape().as_list()[-1]
        
        lhs_2 = linear(lhs, dim, True, bias_start=0.0, scope="lhs_2", squeeze=False, wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        rhs_2 = linear(rhs, dim, True, bias_start=0.0, scope="rhs_2", squeeze=False, wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        f = tf.sigmoid(lhs_2 + rhs_2)

        out = f * lhs + (1 - f) * rhs
        return out

def self_attention_layer(config, is_train, p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        self_att = self_attention(config, is_train, p, p_mask)
        # p0 = fuse_gate(config, is_train, p, self_att, scope="p0")
        return self_att

def basic_block(config, p, h, p_mask, h_mask, input_keep_prob=1.0, is_train=None, scope=None):
    N, JX, JQ = tf.shape(p)[0], tf.shape(p)[1], tf.shape(h)[1]
    d = p.get_shape().as_list()[-1]
    aug_p_mask = tf.tile(tf.expand_dims(p_mask, 2), [1, 1, JQ])
    aug_h_mask = tf.tile(tf.expand_dims(h_mask, 1), [1, JX, 1])
    mask = aug_p_mask & aug_h_mask # [N, JX, JQ]
    with tf.variable_scope(scope or "basic_block"):
        with tf.variable_scope("att"):
            aug_p = tf.tile(tf.expand_dims(p, 2), [1, 1, JQ, 1]) # [N, JX, JQ, d]
            aug_h = tf.tile(tf.expand_dims(h, 1), [1, JX, 1, 1]) # [N, JX, JQ, d]
            
            similarity = get_logits([aug_p, aug_h], None, True, 
                                    wd=config.wd, mask=mask, 
                                    is_train=is_train, 
                                    func=config.att_func, 
                                    scope='similarity') # [N, JX, JQ]

            h_att = softsel(aug_h, similarity, mask=mask, scope='h_att') # [N, JX, d]
            p_att = softsel(tf.transpose(aug_p, [0, 2, 1, 3]), # [N, JQ, JX, d]
                            tf.transpose(similarity, [0, 2, 1]), 
                            mask=tf.transpose(mask, [0, 2, 1]), 
                            scope='p_att') # [N, JQ, d]

        with tf.variable_scope("self_att"):
            p = self_attention_layer(config, is_train, p, p_mask=p_mask, scope="self_attention") # [N, JX, d]    
            tf.get_variable_scope().reuse_variables()
            h = self_attention_layer(config, is_train, h, p_mask=h_mask, scope="self_attention") # [N, JQ, d]    
        
        p = tf.concat([p * h_att, tf.abs(p - h_att)], -1) # [N, JX, d]
        h = tf.concat([h * p_att, tf.abs(h - p_att)], -1) # [N, JQ, d]

        with tf.variable_scope("projecion"):
            p = F(p, d, activation=tf.nn.relu, scope='p', input_keep_prob=config.input_keep_prob, is_train=is_train, wd=config.wd)
            tf.get_variable_scope().reuse_variables()
            h = F(h, d, activation=tf.nn.relu, scope='p', input_keep_prob=config.input_keep_prob, is_train=is_train)

        with tf.variable_scope("rnn"):
            p_len = tf.reduce_sum(tf.cast(p_mask, tf.int32), 1)
            h_len = tf.reduce_sum(tf.cast(h_mask, tf.int32), 1)
            
            ph, _ = rnn(config.rnn_type, tf.concat([p, h], 0), p_len, config.hidden_size, scope='p_rnn', 
                    dropout_keep_prob=config.input_keep_prob, is_train=is_train, wd=config.wd)
            p, h = tf.split(ph, 2)

            # p, _ = rnn(config.rnn_type, p, p_len, config.hidden_size, scope='p_rnn', 
            #             dropout_keep_prob=config.input_keep_prob, is_train=is_train, wd=config.wd)
            # tf.get_variable_scope().reuse_variables()
            # h, _ = rnn(config.rnn_type, h, h_len, config.hidden_size, scope='p_rnn', 
            #             dropout_keep_prob=config.input_keep_prob, is_train=is_train)
            p = p * tf.to_float(tf.expand_dims(p_mask, -1)) # [N, JX, d]
            h = h * tf.to_float(tf.expand_dims(h_mask, -1)) # [N, JQ, d]

        p_max = tf.reduce_max(p, 1)
        h_max = tf.reduce_max(h, 1)

        attentive = True
        if attentive:
            p_att_logits = linear(p, 1, True, scope='p_att_logits', squeeze=True, 
                                input_keep_prob=config.input_keep_prob, is_train=is_train, wd=config.wd)
            p_att = softsel(p, p_att_logits, mask=p_mask, scope='p_att')
            tf.get_variable_scope().reuse_variables()
            h_att_logits = linear(h, 1, True, scope='p_att_logits', squeeze=True, 
                                input_keep_prob=config.input_keep_prob, is_train=is_train)
            h_att = softsel(h, h_att_logits, mask=h_mask, scope='h_att')

            p_max = tf.concat([p_max, p_att], -1)
            h_max = tf.concat([h_max, h_att], -1)

    return p, h, p_max, h_max
