import os
import numpy as np
import random
import re
import scipy
import csv
import io
import json
from tqdm import tqdm
import logging
from collections import Counter
import pickle
from nltk.corpus import stopwords
import string



LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1
}

class Dataset(object):
    """docstring for Dataset"""
    def process(self, path, is_train=False, shuffle=False):
        pass

    def word_iter(self, set_name=None, char_level=False):
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            if char_level:
                for sample in data_set:
                    for token in sample[0][1]:
                        for char in token:
                            yield char
                    for token in sample[1][1]:
                        for char in token:
                            yield char
            else:
                for sample in data_set:
                    for token in sample[0][0]:
                        yield token
                    for token in sample[1][0]:
                        yield token

    def prepare_data(self, seqs, char_seqs=None, set_name='train'):
        lengths = [len(seq) for seq in seqs]
        n_samples = len(seqs)
        # max_len = np.max(lengths)
        max_len = self.max_seq_len
        x = np.zeros((n_samples, max_len)).astype('int32')
        x_mask = np.zeros((n_samples, max_len), dtype='float')
        for idx, seq in enumerate(seqs):
            x[idx, :lengths[idx]] = seq
            x_mask[idx, :lengths[idx]] = 1.0
            
        max_word_len = self.cfg.max_word_len
        cx = np.zeros((n_samples, max_len, max_word_len)).astype('int32')
        for idx, char_seq in enumerate(char_seqs):
            for w_idx, word in enumerate(char_seq):
                word_length = min(max_word_len, len(word))
                cx[idx, w_idx, : word_length] = word[: word_length]

        return x, cx, x_mask

    def one_mini_batch(self, data, indices, pad_id, set_name):
        assert pad_id == 0
        mb_x1 = [data[i][0][0] for i in indices]
        mb_cx1 = [data[i][0][1] for i in indices]
        mb_x2 = [data[i][1][0] for i in indices]
        mb_cx2 = [data[i][1][1] for i in indices]
        max_len1 = np.max([len(seq) for seq in mb_x1])
        max_len2 = np.max([len(seq) for seq in mb_x2])
        self.max_seq_len = np.max([max_len1, max_len2])
        mb_x1, mb_cx1, mb_mask1 = self.prepare_data(mb_x1, mb_cx1, set_name=set_name)
        mb_x2, mb_cx2, mb_mask2 = self.prepare_data(mb_x2, mb_cx2, set_name=set_name)
        
        mb_y = [data[i][2] for i in indices]
        new_mb_y = np.zeros((mb_x1.shape[0], self.cfg.pred_size), 'int')
        new_mb_y[np.arange(mb_x1.shape[0]), mb_y] = 1
        mb_y = new_mb_y
        
        batch_data = ((mb_x1, mb_cx1, mb_mask1), (mb_x2, mb_cx2, mb_mask2), mb_y)
        
        return batch_data

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True, num_batches=None):
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        if num_batches is not None:
            indices = indices[0: num_batches * batch_size]
            data_size = len(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            if len(batch_indices) == batch_size:
                yield self.one_mini_batch(data, batch_indices, pad_id, set_name=set_name)

    def convert_to_ids(self, vocab, char_vocab=None):
        self.train_set = self.conver_to_ids_for_dataset('train', vocab, char_vocab=char_vocab) 
        self.dev_set = self.conver_to_ids_for_dataset('dev', vocab, char_vocab=char_vocab)
        self.test_set = self.conver_to_ids_for_dataset('test', vocab, char_vocab=char_vocab)


    def conver_to_ids_for_dataset(self, set_name, vocab, char_vocab=None):
        if set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

        if self.cfg.debug:
            # import ipdb; ipdb.set_trace()
            data_set = data_set[:128]

        new_data_set = []
        filter_some_samples = True
        if filter_some_samples:
            self.logger.info("filtering some data samples...")
        n_filtered_sample = 0
        for sample in data_set:
            premise_ids = vocab.convert_to_ids(sample[0][0])
            hypothesis_ids = vocab.convert_to_ids(sample[1][0])

            premise_char_ids, hypothesis_char_ids = [], []
            for word in sample[0][1]:
                premise_char_ids.append(char_vocab.convert_to_ids(word))
            for word in sample[1][1]:
                hypothesis_char_ids.append(char_vocab.convert_to_ids(word))

            if filter_some_samples and set_name == 'train':
                if len(premise_ids) > self.max_seq_len or len(hypothesis_ids) > self.max_seq_len:
                    n_filtered_sample += 1
                    continue
            new_data_set += [((premise_ids, premise_char_ids), (hypothesis_ids, hypothesis_char_ids), sample[2])]
        
        self.logger.info("filtered {} samples".format(n_filtered_sample))
        return new_data_set
                   
class SNLIDataSet(Dataset):
    """docstring for SNLIDataSet"""
    def __init__(self, cfg, data_dir):
        super(SNLIDataSet, self).__init__()
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.model_name)

        self.max_word_len = self.cfg.max_word_len
        self.max_seq_len = self.cfg.max_seq_len

        # import ipdb; ipdb.set_trace()


        if os.path.exists(os.path.join(data_dir, "train_set.json")) \
        and os.path.exists(os.path.join(data_dir, "dev_set.json")) \
        and os.path.exists(os.path.join(data_dir, "test_set.json")):
            self.train_set = json.load(open(os.path.join(data_dir, "train_set.json"), 'r'))
            self.dev_set = json.load(open(os.path.join(data_dir, "dev_set.json"), 'r'))
            self.test_set = json.load(open(os.path.join(data_dir, "test_set.json"), 'r'))

        else:
            datapath = "../data"
            train_set_path = "{}/snli_1.0/snli_1.0_train.jsonl".format(datapath)
            dev_set_path = "{}/snli_1.0/snli_1.0_dev.jsonl".format(datapath)
            test_set_path = "{}/snli_1.0/snli_1.0_test.jsonl".format(datapath)

            self.train_set = self.process(train_set_path, is_train=True)
            self.dev_set = self.process(dev_set_path)
            self.test_set = self.process(test_set_path)

            json.dump(self.train_set, open(os.path.join(data_dir, "train_set.json"), 'w'))
            json.dump(self.dev_set, open(os.path.join(data_dir, "dev_set.json"), 'w'))
            json.dump(self.test_set, open(os.path.join(data_dir, "test_set.json"), 'w'))
            
        self.logger.info("Train set Size: {}".format(len(self.train_set)))
        self.logger.info("Dev set Size: {}".format(len(self.dev_set)))
        self.logger.info("Test set Size: {}".format(len(self.test_set)))

    def process(self, path, is_train=False, shuffle=False):
        # import ipdb; ipdb.set_trace()

        def tokenize(string):
            string = re.sub(r'\(|\)', '', string)
            return string.split()


        cfg = self.cfg
        data = []
        premises, hypothesises = [], []
        premises_char, hypothesises_char = [], []
        premises_exact_matches, hypothesises_exact_matches = [], []

        labels = []

        n_samples = 0

        # with open(path, encoding='utf-8') as f:
        with open(path, 'r') as f:
            for line in tqdm(f):
                loaded_example = json.loads(line)
                if loaded_example["gold_label"] not in LABEL_MAP:
                    continue

                n_samples += 1
                if n_samples > 1000 and cfg.debug:
                    break

                ###### word #########
                premise = tokenize(loaded_example['sentence1_binary_parse'])
                hypothesis = tokenize(loaded_example['sentence2_binary_parse'])
                
                premises.append(premise)
                hypothesises.append(hypothesis)
                
                ###### char #########
                premise_char = [list(w) for w in premise]
                hypothesis_char = [list(w) for w in hypothesis]

                premises_char.append(premise_char)
                hypothesises_char.append(hypothesis_char)

                ###### label #########
                labels.append(LABEL_MAP[loaded_example["gold_label"]])


            data = zip(zip(premises, premises_char), zip(hypothesises, hypothesises_char), labels)

            if shuffle:
                random.seed(1)
                random.shuffle(data)
        return data


    '''
    def prepare_data(self, seqs, char_seqs=None, set_name='train'):
        lengths = [len(seq) for seq in seqs]
        n_samples = len(seqs)
        max_len = np.max(lengths)
        x = np.zeros((n_samples, max_len)).astype('int32')
        x_mask = np.zeros((n_samples, max_len), dtype='float')
        for idx, seq in enumerate(seqs):
            x[idx, :lengths[idx]] = seq
            x_mask[idx, :lengths[idx]] = 1.0
            
        max_word_len = self.cfg.max_word_len
        cx = np.zeros((n_samples, max_len, max_word_len)).astype('int32')
        for idx, char_seq in enumerate(char_seqs):
            for w_idx, word in enumerate(char_seq):
                word_length = min(max_word_len, len(word))
                cx[idx, w_idx, : word_length] = word[: word_length]

        return x, cx, x_mask

    def one_mini_batch(self, data, indices, pad_id, set_name):
        assert pad_id == 0
        
        mb_x1 = [data[i][0][0] for i in indices]
        mb_cx1 = [data[i][0][1] for i in indices]
        mb_x1, mb_cx1, mb_mask1 = self.prepare_data(mb_x1, mb_cx1, set_name=set_name)
        mb_x2 = [data[i][1][0] for i in indices]
        mb_cx2 = [data[i][1][1] for i in indices]
        mb_x2, mb_cx2, mb_mask2 = self.prepare_data(mb_x2, mb_cx2, set_name=set_name)
        
        mb_y = [data[i][2] for i in indices]
        new_mb_y = np.zeros((mb_x1.shape[0], self.cfg.pred_size), 'int')
        new_mb_y[np.arange(mb_x1.shape[0]), mb_y] = 1
        mb_y = new_mb_y
        
        batch_data = ((mb_x1, mb_cx1, mb_mask1), (mb_x2, mb_cx2, mb_mask2), mb_y)
        
        return batch_data

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True, num_batches=None):
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        if num_batches is not None:
            indices = indices[0: num_batches * batch_size]
            data_size = len(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            if len(batch_indices) == batch_size:
                yield self.one_mini_batch(data, batch_indices, pad_id, set_name=set_name)

    def convert_to_ids(self, vocab, char_vocab=None):
        self.train_set = self.conver_to_ids_for_dataset('train', vocab, char_vocab=char_vocab) 
        self.dev_set = self.conver_to_ids_for_dataset('dev', vocab, char_vocab=char_vocab)
        self.test_set = self.conver_to_ids_for_dataset('test', vocab, char_vocab=char_vocab)


    def conver_to_ids_for_dataset(self, set_name, vocab, char_vocab=None):
        if set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

        if self.cfg.debug:
            # import ipdb; ipdb.set_trace()
            data_set = data_set[:128]

        new_data_set = []
        filter_some_samples = True
        if filter_some_samples:
            self.logger.info("filtering some data samples...")
        n_filtered_sample = 0
        for sample in data_set:
            premise_ids = vocab.convert_to_ids(sample[0][0])
            hypothesis_ids = vocab.convert_to_ids(sample[1][0])

            premise_char_ids, hypothesis_char_ids = [], []
            for word in sample[0][1]:
                premise_char_ids.append(char_vocab.convert_to_ids(word))
            for word in sample[1][1]:
                hypothesis_char_ids.append(char_vocab.convert_to_ids(word))

            if filter_some_samples and set_name == 'train':
                if len(premise_ids) > self.max_seq_len or len(hypothesis_ids) > self.max_seq_len:
                    n_filtered_sample += 1
                    continue
            new_data_set += [((premise_ids, premise_char_ids), (hypothesis_ids, hypothesis_char_ids), sample[2])]
        
        self.logger.info("filtered {} samples".format(n_filtered_sample))
        return new_data_set
    '''
        
class MultiNLIDataSet(Dataset):
    """docstring for MultiNLIDataSet"""
    def __init__(self, cfg, data_dir):
        super(MultiNLIDataSet, self).__init__()
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.model_name)

        self.max_word_len = self.cfg.max_word_len
        self.max_seq_len = self.cfg.max_seq_len

        # import ipdb; ipdb.set_trace()

        if os.path.exists(os.path.join(data_dir, "train_set.json")) \
        and os.path.exists(os.path.join(data_dir, "dev_set.json")) \
        and os.path.exists(os.path.join(data_dir, "test_set.json")):
            self.train_set = json.load(open(os.path.join(data_dir, "train_set.json"), 'r'))
            self.dev_set = json.load(open(os.path.join(data_dir, "dev_set.json"), 'r'))
            self.test_set = json.load(open(os.path.join(data_dir, "test_set.json"), 'r'))

        else:
            datapath = "../data"
            train_set_path = "{}/multinli_0.9/multinli_0.9_train.jsonl".format(datapath)
            dev_set_path = "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(datapath)
            test_set_path = "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(datapath)

            self.train_set = self.process(train_set_path, is_train=True)
            self.dev_set = self.process(dev_set_path)
            self.test_set = self.process(test_set_path)

            json.dump(self.train_set, open(os.path.join(data_dir, "train_set.json"), 'w'))
            json.dump(self.dev_set, open(os.path.join(data_dir, "dev_set.json"), 'w'))
            json.dump(self.test_set, open(os.path.join(data_dir, "test_set.json"), 'w'))
            
        self.logger.info("Train set Size: {}".format(len(self.train_set)))
        self.logger.info("Dev set Size: {}".format(len(self.dev_set)))
        self.logger.info("Test set Size: {}".format(len(self.test_set)))

    def process(self, path, is_train=False, shuffle=False):
        # import ipdb; ipdb.set_trace()

        def tokenize(string):
            string = re.sub(r'\(|\)', '', string)
            return string.split()

        cfg = self.cfg
        data = []
        premises, hypothesises = [], []
        premises_char, hypothesises_char = [], []
        premises_exact_matches, hypothesises_exact_matches = [], []

        labels = []

        n_samples = 0

        # with open(path, encoding='utf-8') as f:
        with open(path, 'r') as f:
            for line in tqdm(f):
                loaded_example = json.loads(line)
                if loaded_example["gold_label"] not in LABEL_MAP:
                    continue

                n_samples += 1
                if n_samples > 1000 and cfg.debug:
                    break

                ###### word #########
                premise = tokenize(loaded_example['sentence1_binary_parse'])
                hypothesis = tokenize(loaded_example['sentence2_binary_parse'])
                
                premises.append(premise)
                hypothesises.append(hypothesis)
                
                ###### char #########
                premise_char = [list(w) for w in premise]
                hypothesis_char = [list(w) for w in hypothesis]

                premises_char.append(premise_char)
                hypothesises_char.append(hypothesis_char)

                ###### label #########
                labels.append(LABEL_MAP[loaded_example["gold_label"]])


            data = zip(zip(premises, premises_char), zip(hypothesises, hypothesises_char), labels)

            if shuffle:
                random.seed(1)
                random.shuffle(data)
        return data


class QuoraDataSet(Dataset):
    """docstring for QuoraDataSet"""
    def __init__(self, cfg, data_dir):
        super(QuoraDataSet, self).__init__()
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.model_name)

        self.max_word_len = self.cfg.max_word_len
        self.max_seq_len = self.cfg.max_seq_len

        # import ipdb; ipdb.set_trace()

        if os.path.exists(os.path.join(data_dir, "train_set.json")) \
        and os.path.exists(os.path.join(data_dir, "dev_set.json")) \
        and os.path.exists(os.path.join(data_dir, "test_set.json")):
            self.train_set = json.load(open(os.path.join(data_dir, "train_set.json"), 'r'))
            self.dev_set = json.load(open(os.path.join(data_dir, "dev_set.json"), 'r'))
            self.test_set = json.load(open(os.path.join(data_dir, "test_set.json"), 'r'))

        else:
            datapath = "../data"
            train_set_path = "{}/Quora_question_pair_partition/train.tsv".format(datapath)
            dev_set_path = "{}/Quora_question_pair_partition/dev.tsv".format(datapath)
            test_set_path = "{}/Quora_question_pair_partition/test.tsv".format(datapath)

            self.train_set = self.process(train_set_path, is_train=True)
            self.dev_set = self.process(dev_set_path)
            self.test_set = self.process(test_set_path)

            json.dump(self.train_set, open(os.path.join(data_dir, "train_set.json"), 'w'))
            json.dump(self.dev_set, open(os.path.join(data_dir, "dev_set.json"), 'w'))
            json.dump(self.test_set, open(os.path.join(data_dir, "test_set.json"), 'w'))
            
        self.logger.info("Train set Size: {}".format(len(self.train_set)))
        self.logger.info("Dev set Size: {}".format(len(self.dev_set)))
        self.logger.info("Test set Size: {}".format(len(self.test_set)))

    def process(self, path, is_train=False, shuffle=False):

        def tokenize(string):
            return string.split()

        cfg = self.cfg
        data = []
        first_sentences, second_sentences = [], []
        first_sentences_char, second_sentences_char = [], []
        labels = []

        # with io.open(path, 'r', encoding='utf-8') as raw_data:
        with open(path, 'r') as raw_data:
            raw_data = csv.reader(raw_data , delimiter='\t')
            for line in tqdm(raw_data):

                ##### word #####
                first_sentence = tokenize(line[1].decode('utf-8'))
                second_sentence = tokenize(line[2].decode('utf-8'))
                first_sentences.append(first_sentence)
                second_sentences.append(second_sentence)

                ##### char #####
                first_sentence_char = [list(w) for w in first_sentence]
                second_sentence_char = [list(w) for w in second_sentence]
                first_sentences_char.append(first_sentence_char)
                second_sentences_char.append(second_sentence_char)

                ##### label #####
                labels.append(int(line[0]))

        data = zip(zip(first_sentences, first_sentences_char), zip(second_sentences, second_sentences_char), labels)

        if shuffle:
            random.seed(1)
            random.shuffle(data)
        return data


class BaiduZhidao(Dataset):
    """docstring for BaiduZhidao"""
    def __init__(self, cfg, data_dir):
        super(BaiduZhidao, self).__init__()
        self.cfg = cfg