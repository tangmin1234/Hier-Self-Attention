import argparse
import logging


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    
    # Basics
    parser.add_argument("-device_type", type=str, default='gpu')
    parser.add_argument("-gpu_id", type=int, default=0)
    parser.add_argument("-gpu_allocate_rate", type=float, default=0.98)
    parser.add_argument("-model_name", type=str, default='RCN')
    parser.add_argument("-dataset_name", type=str, default='snli')
    parser.add_argument("-pred_size", type=int, default=3)
    parser.add_argument('-debug', type='bool', default=False, help='whether it is debug mode')
    parser.add_argument('-tune_embedding', type='bool', default=False, help="fine tune embedding")
    parser.add_argument('-test_only', type='bool', default=False, help='test_only: no need to run training process')
    parser.add_argument('-random_seed', type=int, default=1000, help='Random seed')
    
    # result file
    parser.add_argument('-correct_result_file', type=str, default="correct_result.txt", help='correct result file')
    parser.add_argument('-wrong_result_file', type=str, default="wrong_result.txt", help='wrong result file')
    parser.add_argument('-model_file', type=str, default='my_model.pkl.gz', help='Model file to save')
    parser.add_argument('-log_file', type=str, default='',  help='Log file')
    
    # Data file
    parser.add_argument('-embedding_file', type=str, default="/home/tangmin/data/glove")
    parser.add_argument("-embed_dim", type=int, default=300)
    parser.add_argument("-fasttext", type='bool', default=False)
    
    # Model details
    parser.add_argument('-embedding_size',type=int, default=None, help='Default embedding size if embedding_file is not given')
    parser.add_argument('-hidden_size',type=int, default=150, help='Hidden size of RNN units')
    parser.add_argument('-num_layers',type=int, default=4, help='Number of RNN layers')
    parser.add_argument("-num_rnn_layers", type=int, default=1)
    parser.add_argument("-num_highway_layers", type=int, default=1)
    parser.add_argument('-rnn_type', type=str, default='bigru', help='RNN type: lstm or gru (default)')
    parser.add_argument('-att_func',type=str,default='tri_linear')
    parser.add_argument('-highway', type='bool', default=True)

    # Optimization details
    parser.add_argument('-batch_size',type=int, default=32)
    parser.add_argument('-num_epoches',type=int, default=100)
    parser.add_argument('-eval_iter',type=int, default=1000)
    parser.add_argument("-print_iter", type=int, default=200)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-grad_clipping',type=float, default=10.0)
    parser.add_argument("-load_model", type='bool', default=False)
    parser.add_argument("-load_step", type=int, default=0)
    parser.add_argument("-char_emb_dim", type=int, default=8)
    parser.add_argument("-input_keep_prob", type=float, default=0.8)
    parser.add_argument("-lower", type='bool', default=True)
    parser.add_argument("-max_seq_len", type=int, default=40)
    parser.add_argument("-max_word_len", type=int, default=16)
    parser.add_argument("-use_additional_features", type='bool', default=False)

    # alation
    parser.add_argument("-use_char_emb", type='bool', default=True)
    parser.add_argument("-char_out_size", type=int, default=100)
    parser.add_argument("-out_channel_dims", type=str, default="20,20,20,20,20")
    parser.add_argument("-filter_heights", type=str, default="1,2,3,4,5")
    parser.add_argument("-share_cnn_weights", type='bool', default=True)
    parser.add_argument("-share_rnn_weights", type='bool', default=True)
    parser.add_argument("-wd", type=float, default=0.0)

    return parser.parse_args()

def get_logger(config):
    logger = logging.getLogger('{}'.format(config.model_name))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    if config.log_file != '':
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger