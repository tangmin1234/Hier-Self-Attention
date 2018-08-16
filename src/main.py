import os
import logging
import pickle
import time
import tensorflow as tf
from config import get_args, get_logger
from rcn import Model
from trainer import Trainer
from dataset import SNLIDataSet, MultiNLIDataSet, QuoraDataSet, BaiduZhidao
from vocab import Vocab


def prepare(cfg, vocab_dir, data_dir):
    logger = logging.getLogger(cfg.model_name)
    if cfg.dataset_name == 'snli':
        snli_data = SNLIDataSet(cfg, data_dir)
    elif cfg.dataset_name == 'mnli':
        snli_data = MultiNLIDataSet(cfg, data_dir)
    elif cfg.dataset_name == 'quora':
        snli_data = QuoraDataSet(cfg, data_dir)
    elif cfg.dataset_name == 'bdzd':
        snli_data = BaiduZhidao(cfg, data_dir)
    else:
        raise ValueError("No such a dataset, dataset name is {}".format(cfg.dataset_name))

    logger.info('Building vocabulary...')
    vocab = Vocab(cfg)
    logger.info("Add tokens into vocab...")
    for word in snli_data.word_iter('train'):
        vocab.add(word)
    for word in snli_data.word_iter('dev'):
        vocab.add(word)
    for word in snli_data.word_iter('test'):
        vocab.add(word)
    
    filter_vocab = True
    if filter_vocab:
        min_cnt = 1
        logger.info('Filter word with frequency {}'.format(min_cnt))
        unfiltered_vocab_size = vocab.size()
        vocab.filter_tokens_by_cnt(min_cnt=min_cnt)
        filtered_num = unfiltered_vocab_size - vocab.size()
        logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.load_pretrained_embeddings(cfg.embedding_file)
    logger.info('Saving vocab...')
    with open(os.path.join(vocab_dir, 'vocab_{}d.voc'.format(cfg.embed_dim)), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Building character vocabulary...')
    char_vocab = Vocab(cfg)
    for char in snli_data.word_iter('train', char_level=True):
        char_vocab.add(char)
    for char in snli_data.word_iter('dev', char_level=True):
        char_vocab.add(char)
    for char in snli_data.word_iter('test', char_level=True):
        char_vocab.add(char)
    unfiltered_char_vocab_size = char_vocab.size()
    char_vocab.filter_tokens_by_cnt(min_cnt=4)
    filtered_num = unfiltered_char_vocab_size - char_vocab.size()
    logger.info('After filter {} char(s), the final character vocab size is {}'.format(filtered_num, char_vocab.size()))

    logger.info('Assigning embeddings...')
    char_vocab.randomly_init_embeddings(embed_dim=cfg.char_emb_dim)
    logger.info('Saving character vocab')
    with open(os.path.join(vocab_dir, 'char_vocab_{}d.voc'.format(cfg.char_emb_dim)), 'wb') as fout:
        pickle.dump(char_vocab, fout)
    
    logger.info('Done with preparing!')


def check_directories(cfg):
    data_obj_dir = os.path.join("../obj", cfg.dataset_name)
    if not os.path.exists(data_obj_dir):
        os.mkdir(data_obj_dir)

    vocab_dir = os.path.join(data_obj_dir, 'vocab')
    data_dir = os.path.join(data_obj_dir, 'data')
    model_dir = os.path.join(data_obj_dir, 'model')

    if not os.path.exists(vocab_dir): 
        os.mkdir(vocab_dir)
    if not os.path.exists(data_dir): 
        os.mkdir(data_dir)
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)

    return vocab_dir, data_dir, model_dir

def main(cfg):
    # import ipdb; ipdb.set_trace()
    logger = get_logger(cfg)
    
    vocab_dir, data_dir, model_dir = check_directories(cfg)

    if cfg.fasttext:
        vocab_dir = os.path.join(vocab_dir, 'fasttext')
    else:
        vocab_dir = os.path.join(vocab_dir, 'glove')
    if not os.path.exists(vocab_dir):
        os.mkdir(vocab_dir)
    
    if not os.path.exists(os.path.join(vocab_dir, 'vocab_{}d.voc'.format(cfg.embed_dim))) \
    or not os.path.join(vocab_dir, 'char_vocab_{}d.voc'.format(cfg.char_emb_dim)):
        prepare(cfg, vocab_dir, data_dir)

    logger.info('Load vocab...')
    with open(os.path.join(vocab_dir, 'vocab_{}d.voc'.format(cfg.embed_dim)), 'rb') as fin1:
        vocab = pickle.load(fin1)
        cfg.vocab_size = vocab.size()
        cfg.embedding_size = vocab.embed_dim
        
    with open(os.path.join(vocab_dir, 'char_vocab_{}d.voc'.format(cfg.char_emb_dim)), 'rb') as fin2:
        char_vocab = pickle.load(fin2)
        cfg.char_vocab_size = char_vocab.size()

    logger.info(cfg)
    cfg.embeddings = vocab.embeddings
    cfg.char_embeddings = char_vocab.embeddings

    logger.info('Initialize the model and trainer...')
    with tf.device("/device:{}:{}".format(cfg.device_type, cfg.gpu_id)):
        model = Model(cfg)
        trainer = Trainer(cfg, model)

    start_time = time.time()

    logger.info("Load dataset...")
    
    if cfg.dataset_name == 'snli':
        snli_data = SNLIDataSet(cfg, data_dir)
    elif cfg.dataset_name == 'mnli':
        snli_data = MultiNLIDataSet(cfg, data_dir)
    elif cfg.dataset_name == 'quora':
        snli_data = QuoraDataSet(cfg, data_dir)
    elif cfg.dataset_name == 'bdzd':
        snli_data = BaiduZhidao(cfg, data_dir)
    else:
        raise ValueError("No such a dataset, dataset name is {}".format(cfg.dataset_name))

    logger.info('Converting text into ids...')
    snli_data.convert_to_ids(vocab, char_vocab)
   
    logger.info("collapsed time {} for loading data.".format(time.time() - start_time))

    logger.info('Creating session')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_allocate_rate)
    gpu_options.allow_growth = True
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())

    if cfg.load_model:
        logger.info("Restoring the model")
        model.restore(sess, model_dir)
        logger.info("Restored!")

    if not cfg.test_only:
        logger.info('Training the model')
        trainer.train(sess, snli_data, model_dir)

    logger.info('Testing the model')
    trainer.dump_answer(sess, snli_data, vocab, evaluate_dataset='test')
    logger.info('work done!')

if __name__ == '__main__':
    cfg = get_args()
    if cfg.embed_dim == 100:
        cfg.embedding_file = os.path.join(cfg.embedding_file, "glove.6B.100d.txt")
    else:
        if cfg.fasttext:
            cfg.embedding_file = os.path.join("/home/tangmin/data/word2vec/fasttext", "wiki-news-300d-1M.vec")
        else:   
            cfg.embedding_file = os.path.join(cfg.embedding_file, "glove.840B.300d.txt")
    if cfg.use_char_emb:
        cfg.char_embedding_file = os.path.join("/home/tangmin/data/glove", "glove.840B.300d-char.txt")

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(cfg.gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main(cfg)
