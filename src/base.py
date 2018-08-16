import time
import logging
import os
import numpy as np
import tensorflow as tf


class ModelBase(object):
    """docstring for ModelBase"""
    def __init__(self, cfg):
        tf.set_random_seed(cfg.random_seed)
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.model_name)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                            initializer=tf.constant_initializer(0),
                                            trainable=False)
        self.loss = None
        self.logits = None
        self.yp = None

        self.addPlaceholder()
        self.buildModel()
        self.buildObjective()
        # self.printNumOfParameter()

        self.summary = None
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        
    def addPlaceholder(self):
        cfg = self.cfg
        self.premise_x = tf.placeholder(tf.int32, [None, None], name='premise')
        self.premise_mask = tf.placeholder(tf.bool, [None, None], name='premise_mask')
        self.hypothesis_x = tf.placeholder(tf.int32, [None, None], name='hypothesis')
        self.hypothesis_mask = tf.placeholder(tf.bool, [None, None], name='hypothesis_mask')
        self.premise_pos = tf.placeholder(tf.int32, [None, None, 47], name='premise_pos')
        self.hypothesis_pos = tf.placeholder(tf.int32, [None, None, 47], name='hypothesis_pos')
        self.premise_char = tf.placeholder(tf.int32, [None, None, cfg.max_word_len], name='premise_char')
        self.hypothesis_char = tf.placeholder(tf.int32, [None, None, cfg.max_word_len], name='hypothesis_char')
        self.premise_exact_match = tf.placeholder(tf.int32, [None, None,1], name='premise_exact_match')
        self.hypothesis_exact_match = tf.placeholder(tf.int32, [None, None,1], name='hypothesis_exact_match')
        self.y = tf.placeholder(tf.int32, [None, None], 'y')
        self.is_train = tf.placeholder(tf.bool, [], 'is_train')
        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
        
    def buildObjective(self):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
        loss = tf.reduce_mean(loss)
        tf.add_to_collection("losses", loss)
        self.loss = tf.add_n(tf.get_collection('losses'), name='loss')

    def get_feed_dict(self, batch_data, learning_rate=None, mode='train'):
        feed_dict = {}
        if mode == 'train':
            feed_dict[self.is_train] = np.array(True)
            feed_dict[self.learning_rate] = np.array(learning_rate)
        else:
            feed_dict[self.is_train] = np.array(False)
        
        feed_dict[self.y] = batch_data[2]
        
        feed_dict[self.premise_x] = batch_data[0][0]
        feed_dict[self.premise_char] = batch_data[0][1]
        feed_dict[self.premise_mask] = batch_data[0][2]
        feed_dict[self.hypothesis_x] = batch_data[1][0]
        feed_dict[self.hypothesis_char] = batch_data[1][1]
        feed_dict[self.hypothesis_mask] = batch_data[1][2]
      
        return feed_dict

    def save(self, sess, model_dir):
        cfg = self.cfg
        save_path = os.path.join(model_dir, cfg.model_file)
        self.saver.save(sess, save_path, global_step=self.global_step)

    def restore(self, sess, model_dir):
        cfg = self.cfg
        if cfg.load_step > 0:
            save_path = "{}-{}".format(os.path.join(model_dir, cfg.model_file), cfg.load_step)
        else:
            checkpoint = tf.train.get_checkpoint_state(model_dir)
            assert checkpoint is not None, "cannot load checkpoint at {}".format(model_dir)
            save_path = checkpoint.model_checkpoint_path
        self.saver.restore(sess, save_path)

   
    def printNumOfParameter(self):
        total_parameters = 0
        for v in tf.trainable_variables():
            if not v.name.endswith("weights:0") and not v.name.endswith("biases:0") and not v.name.endswith('kernel:0') and not v.name.endswith('bias:0'):
                continue
            shape = v.get_shape().as_list()
            param_num = 1
            for dim in shape:
                param_num *= dim
            total_parameters += param_num
        self.logger.info("total trainable parameters = {}".format(total_parameters))
