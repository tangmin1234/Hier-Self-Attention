import os
import json
import numpy as np

class Vocab(object):
    """docstring for Vocab"""
    def __init__(self, cfg):
        super(Vocab, self).__init__()
        self.cfg = cfg
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = self.cfg.lower

        self.embed_dim = None
        self.embedding = None
        
        self.pad_token = '<blank>'
        self.unk_token = '<unk>'
        self.initial_tokens = []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

    def size(self):
        return len(self.id2token)

    def get_id(self, token):
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx

        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def randomly_init_embeddings(self, embed_dim):
        self.embed_dim = embed_dim
        self.embeddings = np.random.uniform(low=-0.25, high=0.25, size=(self.size(), self.embed_dim))
        for token in [self.pad_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

    def filter_tokens_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def load_pretrained_embeddings(self, embedding_path):
        # import ipdb; ipdb.set_trace()
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split()
                token = contents[0].decode('utf8')
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
        self.embed_dim = self.cfg.embed_dim
        filtered_tokens = trained_embeddings.keys()
        print("{} tokens were found in pretrained embeddings...".format(len(filtered_tokens)))
        # self.token2id = {}
        # self.id2token = {}
        # for token in self.initial_tokens:
        #     self.add(token, cnt=0)
        # for token in filtered_tokens:
        #     self.add(token, cnt=0)
        padding_embedding = np.zeros([2, self.embed_dim])
        self.embeddings = np.random.uniform(low=-0.25, high=0.25, size=(self.size()-2, self.embed_dim))
        # self.embeddings = np.zeros([self.size()-2, self.embed_dim])
        self.embeddings = np.concatenate((padding_embedding, self.embeddings), axis=0)
        
        count = 0
        for token in self.token2id.keys():
            if token in filtered_tokens:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]
                count += 1

        assert count == len(filtered_tokens)

    def convert_to_ids(self, tokens, type='word'):
        vec = [self.get_id(token) for token in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens