from hyperopt import hp

affinity_vals = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
linkage_vals = ['complete', 'average', 'single']
init_vals = ["k-means++", "random"]
algorithm_vals = ['auto', 'elkan']


kmeans_search_space = [{
    'init': hp.choice('init', init_vals),
    'algorithm': hp.choice('algorithm', algorithm_vals),
    'n_init': hp.quniform('n_init', 10, 100, 2),
    'max_iter': hp.quniform('max_iter', 100, 10000, 100),
}]

kmeans_int_params = ['n_init', 'max_iter']

agglomerative_search_space = [{
    'affinity': hp.choice('affinity', affinity_vals),
    'linkage': hp.choice('linkage', linkage_vals),
}]


dbscan_search_space = [{
  'eps': hp.uniform('eps', 0.001, 0.999),
  'min_samples': hp.quniform('min_samples', 1,10, 1)
}]

dbscan_int_params = ['min_samples']

fasttext_search_space = [{
    'model': hp.choice('model', ["cbow", "skipgram"]),
    'lr': hp.uniform('lr',0.00001,0.1),
    'dim': hp.quniform('dim', 10, 250, 5),
    'ws': hp.quniform('ws', 1, 10, 1),
    'minCount': hp.quniform('minCount', 1, 10, 1),
    'minn': hp.quniform('minn', 1, 3, 1),
    'maxn': hp.quniform('maxn', 3, 6, 1),
    'neg': hp.quniform('neg', 1, 20, 1),
    'wordNgrams': hp.quniform('wordNgrams', 1, 5, 1),
    'loss': hp.choice('loss', ["ns", "hs", "softmax", "ova"])
}]

glove_search_space = [{
    'vocab_min_count': hp.quniform('vocab_min_count', 1, 20, 1),
    'vector_size':  hp.quniform('vector_size', 10, 250, 5),
    'max_iter': hp.quniform('max_iter', 10, 20, 1),
    'window_size': hp.quniform('window_size', 1, 30, 1),
    'x_max': hp.quniform('x_max', 1, 100, 2),
}]

bert_search_space = [{
    'attention_probs_dropout_prob': hp.quniform('vocab_min_count', 0.01, 0.4, 0.01),
}]

roberta_gpt2_search_space = [{
    'num_train_epochs': hp.quniform('num_training_epochs', 1, 20, 1),
    'learning_rate': hp.quniform('learning_rate',0.00001,0.1, 0.00001), 
    'train_batch_size': hp.quniform('train_batch_size', 4, 64, 4),
    'warmup_steps': hp.quniform('warmup_steps', 1, 5, 1),
    'max_seq_length': hp.quniform('max_seq_length', 32, 256, 4)
}]

doc2vec_search_space = [{
    'vector_size': hp.quniform('vector_size', 10, 250, 5),
    'window': hp.quniform('window', 1, 20, 1)
}]

word2vec_search_space = [{
    'vector_size': hp.quniform('vector_size', 10, 250, 5),
    'window': hp.quniform('window', 1, 20, 1),
    'min_count': hp.quniform('min_count', 1, 20, 1),
    'hs': hp.choice('hs', [0, 1]),
    'alpha': hp.uniform('alpha', 0.00001, 0.1)
}]

sentence_bert_model_choice_lst = ['all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 
    'multi-qa-distilbert-cos-v1', 'all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1', 'paraphrase-albert-small-v2',
    'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-MiniLM-L3-v2', 'distiluse-base-multilingual-cased-v2']

sentence_bert_search_space = [{'pretrained_model': hp.choice('pretrained_model', sentence_bert_model_choice_lst),
                                'batch_size': hp.quniform('batch_size', 256, 1024, 32),
                                'normalize_embeddings': hp.choice('normalize_embeddings', [True, False])}]

glove_pretrained_choice_list = ['glove.6B.50d.txt', 'glove.6B.100d.txt', 'glove.6B.200d.txt', 'glove.6B.300d.txt']

glove_pretrained_search_space = [{'pretrained_model': hp.choice('pretrained_model', glove_pretrained_choice_list)}]

fasttext_int_params = ['dim', 'ws', 'minCount', 'minn', 'maxn', 'neg', 'wordNgrams']
glove_int_params = ['vocab_min_count', 'vector_size', 'max_iter', 'window_size', 'x_max']
roberta_gpt2_int_params = ['num_train_epochs', 'train_batch_size', 'warmup_steps', 'max_seq_length']
doc2vec_int_params = ['window', 'vector_size']
word2vec_int_params = ['vector_size', 'window', 'min_count']
sentence_bert_int_params = ['batch_size']
