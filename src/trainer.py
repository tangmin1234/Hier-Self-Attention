import logging
import time
import os
import tensorflow as tf

from tf_utils import get_optimizer
from general import average_gradients


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, cfg, model):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.model_name)
        self.model = model
        self.loss = model.loss
        self.yp = model.yp
        self.accuracy = model.accuracy
        self.global_step = model.global_step
        self.optimizer = get_optimizer(cfg, self.model.learning_rate)
        self.train_op = self.build_train_op()

    def build_train_op(self):
        cfg = self.cfg
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        if cfg.grad_clipping is not None:
            grads, vars = zip(*grads_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, cfg.grad_clipping)
            grads = zip(grads, vars)
        else:
            grads = grads_and_vars
        
        train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        maintain_averages_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_op]):
            train_op = tf.group(maintain_averages_op)
        return train_op

        
    def train(self, sess, data, model_dir):
        # import ipdb; ipdb.set_trace()
        cfg = self.cfg
        n_updates = 0
        best_acc = 0.0
        learning_rate = cfg.learning_rate

        for epoch in range(cfg.num_epoches):
            train_batches = data.gen_mini_batches('train', cfg.batch_size, 0)
            start_time = time.time()
            for bidx, batch_data in enumerate(train_batches):
                feed_dict = self.model.get_feed_dict(batch_data, learning_rate, mode='train')
                fetches = [self.loss, self.train_op]
                train_loss, _ = sess.run(fetches, feed_dict=feed_dict)
                
                n_updates += 1
                if n_updates % cfg.print_iter == 0:
                    self.logger.info('epoch/iter = {}/{}, loss = {:.2f}, elapsed time = {:.2f} (s)'.format(
                                    epoch, n_updates, train_loss, time.time() - start_time))
                
                if n_updates % cfg.eval_iter == 0:
                    self.logger.info("+" * 60)
                    
                    train_acc, train_loss = self.evaluate(sess, data, evaluate_dataset='train', num_batches=30)
                    self.logger.info("Train loss {:.2f}, Train accuracy {:.4f}".format(train_loss, train_acc))
                    dev_acc, dev_loss = self.evaluate(sess, data, evaluate_dataset='dev', num_batches=None)
                    self.logger.info('Dev loss {:.2f}, Dev accuracy: {:.4f}'.format(dev_loss, dev_acc))
                    test_acc, test_loss = self.evaluate(sess, data, evaluate_dataset='test', num_batches=None)
                    self.logger.info('Test loss {:.2f}, Test accuracy: {:.4f}'.format(test_loss, test_acc))
                    
                    if test_acc > best_acc:
                        best_acc = test_acc
                        self.model.save(sess, model_dir)
                        self.logger.info('epoch = {}, n_updates = {}, dev_acc = {:.4f}, best test accuracy: {:.4f}'.format(
                                            epoch, n_updates, dev_acc, best_acc))
                    
                    self.logger.info("+" * 60)
                        
            learning_rate = self.schedule_lr(epoch, learning_rate)

    def schedule_lr(self, epoch, learning_rate):
        if epoch == 30 or epoch == 40 or epoch == 50:
            learning_rate = learning_rate / 2
            self.logger.info('halving learning rate to {}'.format(learning_rate))
        return learning_rate

    def multi_gpu_train(self):       
        pass

    def evaluate(self, sess, data, evaluate_dataset=None, num_batches=None):
        cfg = self.cfg

        batches = data.gen_mini_batches(evaluate_dataset, cfg.batch_size, 0, num_batches=num_batches)
        
        total_loss, total_acc = 0.0, 0.0
        total_sample, good_prediction_sample, num_batches = 0, 0, 0
        yps = []
        for bidx, batch_data in enumerate(batches):
            feed_dict = self.model.get_feed_dict(batch_data, mode='eval')
            fetches = [self.accuracy, self.yp, self.loss]
            acc, yp, loss = sess.run(fetches, feed_dict=feed_dict)
            total_sample += batch_data[0][0].shape[0]
            num_batches += 1
            good_prediction_sample += acc
            total_loss += loss
            yps.append(yp)
        avg_loss = total_loss / num_batches
        avg_acc = float(good_prediction_sample) / float(total_sample)
        
        return avg_acc, avg_loss

    def dump_answer(self, sess, data, vocab, model_dir, evaluate_dataset=None, num_batches=None):
        cfg = self.cfg
        correct_result_file = os.path.join(model_dir, cfg.correct_result_file)
        correct_result = open(correct_result_file, 'w')
        wrong_result_file = os.path.join(model_dir, cfg.wrong_result_file)
        wrong_result = open(wrong_result_file, 'w')
        batches = data.gen_mini_batches(evaluate_dataset, cfg.batch_size / 2, 0, shuffle=False, num_batches=num_batches)
        total_acc = []
        for bidx, batch_data in enumerate(batches):
            feed_dict = self.model.get_feed_dict(batch_data)
            acc, pred = sess.run([self.accuracy, self.yp], feed_dict=feed_dict)
            total_acc.append(acc)
            (mb_x1, _, mb_mask1), (mb_x2, _, mb_mask2), mb_y = batch_data
            N = mb_y.shape[0]
           
            if len(pred) != mb_y.shape[0]:
                continue
                
            for i, (prediction, true_answer) in enumerate(zip(pred.tolist(), np.argmax(mb_y, axis=1))):
                if prediction == true_answer:
                    correct_result.write("passage: {}\n".format(self.convert_to_raw_text(mb_x1[i], mb_mask1[i], vocab)))
                    correct_result.write("question: {}\n".format(self.convert_to_raw_text(mb_x2[i], mb_mask2[i], vocab)))
                    correct_result.write("options:\t")
                    for j in range(4):
                        correct_result.write("{}: {}\t".format(str(unichr(j+65)), self.convert_to_raw_text(mb_x3[i][j], mb_mask3[i][j], vocab)))
                    correct_result.write("\n")
                    correct_result.write("ground truth: {}          prediction answer: {}\n".format(str(unichr(true_answer+65)), str(unichr(prediction+65))))
                    correct_result.write("\n\n")
                else:
                    wrong_result.write("passage: {}\n".format(self.convert_to_raw_text(mb_x1[i], mb_mask1[i], vocab)))
                    wrong_result.write("question: {}\n".format(self.convert_to_raw_text(mb_x2[i], mb_mask2[i], vocab)))
                    wrong_result.write("options:\t")
                    for j in range(4):
                        wrong_result.write("{}: {}\t".format(str(unichr(j+65)), self.convert_to_raw_text(mb_x3[i][j], mb_mask3[i][j], vocab)))
                    wrong_result.write("\n")
                    wrong_result.write("ground truth: {}            prediction answer: {}\n".format(str(unichr(true_answer+65)), str(unichr(prediction+65))))
                    wrong_result.write("\n\n")

        total_acc = sum(total_acc) / len(total_acc)
        self.logger.info('{} accuracy: {:.4f}'.format(evaluate_dataset, total_acc))
        correct_result.close()
        wrong_result.close()

    def convert_to_raw_text(self, seq, seq_mask, vocab):
        text = None
        if len(seq.shape) == 2:
            seq = [token for tokens in seq.tolist() for token in tokens]
            seq_mask = [token_mask for tokens_mask in seq_mask.tolist() for token_mask in tokens_mask]
            text = [vocab.get_token(token_id) for idx, (token_id, token_id_mask) in enumerate(zip(seq, seq_mask)) if token_id_mask != 0.0]
            text = ' '.join(text)
        elif len(seq.shape) == 1:
            seq_len = np.sum(seq_mask)
            text = [vocab.get_token(token_id) for idx, token_id in enumerate(seq.tolist()) if idx < int(seq_len)]
            text = ' '.join(text)
        else:
            raise NotImplementedError("")
        return text