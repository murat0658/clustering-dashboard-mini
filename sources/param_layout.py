import dash_core_components as dcc
import dash_html_components as html

corporate_colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',
    'dark-green' : 'rgb(57, 81, 85)',
    'medium-green' : 'rgb(93, 113, 120)',
    'light-green' : 'rgb(186, 218, 212)',
    'pink-red' : 'rgb(255, 101, 131)',
    'dark-pink-red' : 'rgb(247, 80, 99)',
    'white' : 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)'
}


def modelParamViewDropDown(id, name, info, values, defaultValue, labels=None):
    if labels == None:
        labels = values
    return html.Label(children = [html.B(name, 
                    title=info, 
                    style={'textDecoration': 'underline', 'cursor': 'pointer', 'color': corporate_colors['dark-green'], 'marginRight': '10px'},
                    ),
                dcc.Dropdown(
                id=id,
                options=[{'label': labels[i], 'value': values[i]} for i in range(len(values))],
                value=defaultValue,
                style={'backgroundColor': corporate_colors['dark-green'], 'width': '15vw', 'marginRight': '10px'}
            )])

def modelParamViewInputBox(id, name, info, _min, _max, step, defaultValue):
    return html.Label(children = [html.B(name, 
                    title=info, 
                    style={'textDecoration': 'underline', 'cursor': 'pointer', 'color': corporate_colors['dark-green'], 'marginRight': '10px'},
                    ),
            html.Br(),
                dcc.Input(
                id=id, type="number", placeholder=str(min)+"-"+str(max),
                min=_min, max=_max, step=step,  value=defaultValue,
                style={'backgroundColor': corporate_colors['dark-green'], 'width': '10vw', 'marginRight': '10px', 'color':'white'}
            )
            ])


fasttext_params = {
    "model": {
        'id': "model", 
        'name': "Model", 
        'info': "unsupervised fasttext model ", 
        'values': ["cbow", "skipgram"], 
        'defaultValue': "skipgram"
    }, "lr": {
        'id': "lr", 
        'name':"Learning Rate", 'info': "Learning Rate",
         '_min':0.00001, 
         '_max':0.1,
         'step': 0.0001, 
         'defaultValue':0.05
    }, 'dim': {
        'id':"dim", 
        'name': "Dimension", 
        'info': "Size of word vectors", 
        '_min': 10 , 
        '_max': 250,
        'step': 5,
        'defaultValue': 50
    }, 'ws': {
        'id': "ws", 
        'name': "Window Size", 
        'info': "Size of the context window", 
        '_min': 1, 
        '_max': 30,
        'step': 1,
        'defaultValue': 5
    }, 'minCount': {
        'id': "minCount", 
        'name': "Min word occurences", 
        'info': "Minimal number of word occurences", 
        '_min': 1, 
        '_max': 10,
        'step': 1,
        'defaultValue': 5
    }, 'minn': {
        'id': "minn", 
        'name':"Min char ngram", 
        'info': "Minimum length of char ngram", 
        '_min': 1, 
        '_max': 3,
        'step': 1,
        'defaultValue': 2
    }, 'maxn': {
        'id': "maxn", 
        'name': "Max char ngram", 
        'info': "Maximum length of char ngram", 
        '_min': 3, 
        '_max': 6,
        'step': 1,
        'defaultValue': 4
    }, 'neg': {
        'id': "neg", 
        'name': "Negs sampled", 
        'info': "Number of negatives sampled(in Negative sampling)", 
        '_min': 1, 
        '_max': 20,
        'step': 1,
        'defaultValue': 5
    }, 'wordNgrams': {
        'id': "wordNgrams", 
        'name': "Word ngram", 
        'info': "max length of word ngram", 
        '_min': 1, 
        '_max': 5,
        'step': 1,
        'defaultValue': 2
    }, 'loss': {
        'id': "loss", 
        'name': "loss fun", 
        'info': "loss function ", 
        'values': ["ns", "hs", "softmax", "ova"], 
        'labels': ["Negative Sampling", "Hierarchical softmax", "softmax", "One-Vs-All"], 
        'defaultValue': "ns"
    }
}


glove_params = {
    'vocab_min_count': {
        'id': "vocab_min_count", 
        'name': "Min Count", 
        'info': "min count of words to include", 
        '_min': 1, 
        '_max': 20,
        'step': 1,
        'defaultValue': 1
    }, 'vector_size': {
        'id': "vector_size", 
        'name': "Vector Size", 
        'info': "Size of word vectors", 
        '_min': 10, 
        '_max': 250,
        'step': 5,
        'defaultValue': 100
    }, 'max_iter': {
        'id': "max_iter", 
        'name': "Max number of iterations", 
        'info': "Maximum number of iterations", 
        '_min': 10, 
        '_max': 20,
        'step': 1,
        'defaultValue': 15
    }, 'window_size': {
        'id': "window_size", 
        'name': "Window size", 
        'info': "Size of the context window", 
        '_min': 1, 
        '_max': 30,
        'step': 1,
        'defaultValue': 15
    }, 'x_max': {
        'id': "x_max", 
        'name': "Max co-occurrences", 
        'info': "maximum number of co-occurrences to use in the weighting function", 
        '_min': 50, 
        '_max': 200,
        'step': 1,
        'defaultValue': 3
    }
   
}

bert_params = {
    'attention_probs_dropout_prob': {
        'id': "attention_probs_dropout_prob", 
        'name': "Dropout Probability", 
        'info': "Attention Probs Dropout Prob", 
        '_min': 0.01, 
        '_max': 0.4,
        'step': 0.01,
        'defaultValue': 0.1
    }
}

roberta_gpt2_params = {
    'num_train_epochs': {
        'id': 'num_train_epochs',
        'name': 'Number of training epochs',
        'info': 'The number of epochs the model will be trained for.',
        '_min': 1,
        '_max': 20,
        'step': 1,
        'defaultValue': 1
    }, 'learning_rate': {
        'id': 'learning_rate',
        'name': 'Learning Rate',
        'info': 'The learning rate for training.',
        '_min': 0.00001,
        '_max': 0.1,
        'step': 0.000001,
        'defaultValue': 0.00001
    }, 'train_batch_size': {
        'id': 'train_batch_size',
        'name': 'Training Batch Size',
        'info': 'The training batch size.',
        '_min': 4,
        '_max': 64,
        'step': 4,
        'defaultValue': 4
    },  'warmup_steps': {
        'id': 'warmup_steps',
        'name': 'Warm-up Steps',
        'info': 'Number of training steps where learning rate will “warm up”.',
        '_min': 0,
        '_max': 5,
        'step': 1,
        'defaultValue': 1
    }, 'max_seq_length': {
        'id': 'max_seq_length',
        'name': 'Maximum Sequence Length',
        'info': 'Maximum sequence length the model will support.',
        '_min': 32,
        '_max': 256,
        'step': 4,
        'defaultValue': 128
    }
}

kmeans_params = {
    'init': {
        'id': 'init', 
        'name': "Init mode", 
        'info': "Method for initialization", 
        'values': ['k-means++', 'random'], 
        'defaultValue': 'random'
    }, 'algorithm': {
        'id': 'algorithm', 
        'name': "algorithm", 
        'info': "Centering the data or not", 
        'values': ['auto', 'elkan'], 
        'defaultValue': 'auto'
    }, 'n_init': {
        'id': 'n_init',
        'name': 'Init points number',
        'info': 'Number of time the k-means algorithm will be run with different centroid seeds',
        '_min': 10,
        '_max': 50,
        'step': 2,
        'defaultValue': 10
    }, 'max_iterint': {
        'id': 'max_iterint',
        'name': 'Max number of iterations',
        'info': 'Maximum number of iterations',
        '_min': 100,
        '_max': 1000,
        'step': 100,
        'defaultValue': 300
    }
}

agglomerative_params = {
    'affinity': {
        'id': 'affinity', 
        'name': "affinity", 
        'info': "Metric used to compute the linkage", 
        'values': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], 
        'defaultValue': 'euclidean'
    }, 'linkage': {
        'id': 'linkage', 
        'name': "linkage", 
        'info': "Metric used to compute the linkage", 
        'values': ['ward', 'complete', 'average', 'single'], 
        'defaultValue': 'ward'
    }
}

doc2vec_params = {
     'vector_size': {
        'id': "vector_size_doc2vec", 
        'name': "Vector Size", 
        'info': "Size of word vectors", 
        '_min': 10, 
        '_max': 250,
        'step': 5,
        'defaultValue': 100
    }, 'window': {
        'id': "window_doc2vec", 
        'name': "Window Size", 
        'info': "Size of the context window", 
        '_min': 1, 
        '_max': 20,
        'step': 1,
        'defaultValue': 5
    }
}

word2vec_params = {
     'vector_size': {
        'id': "vector_size_word2vec", 
        'name': "Vector Size", 
        'info': "Size of word vectors", 
        '_min': 10, 
        '_max': 250,
        'step': 5,
        'defaultValue': 100
    }, 'window': {
        'id': "window_word2vec", 
        'name': "Window Size", 
        'info': "Size of the context window", 
        '_min': 1, 
        '_max': 20,
        'step': 1,
        'defaultValue': 5
    },
    'min_count': {
        'id': "min_count", 
        'name': "Min word freq", 
        'info': "Ignores all words with total frequency lower than this", 
        '_min': 1, 
        '_max': 20,
        'step': 1,
        'defaultValue': 5
    }, 'hs': {
        'id': "hs", 
        'name': "Loss Function(hs, ns)", 
        'info': "1: Hierarchical softmax, 0: negative sampling", 
        'values': [0,1], 
        'defaultValue': 1
    }, 'alpha': {
        'id': "alpha", 
        'name': "alpha",
        'info': "Learning rate",
        '_min': 0.00001,
        '_max': 0.1,
        'step': 0.000001,
        'defaultValue': 0.00001
    }
}


sentence_bert_params = {
     'batch_size': {
        'id': "vector_size_word2vec", 
        'name': "Vector Size", 
        'info': "Size of word vectors", 
        '_min': 4, 
        '_max': 64,
        'step': 4,
        'defaultValue': 8
    }, 'pretrained_model': {
        'id': "pretrained_model", 
        'name': "Pretrained model", 
        'info': "Pretrained model choice", 
        'values': ['all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 
    'multi-qa-distilbert-cos-v1', 'all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1', 'paraphrase-albert-small-v2',
    'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-MiniLM-L3-v2', 'distiluse-base-multilingual-cased-v2'],
        'defaultValue': 'all-MiniLM-L6-v2'
    },
    'normalize_embeddings': {
        'id': "normalize_embeddings", 
        'name': "Normalize or not embeddings", 
        'info': "Normalize or not embeddings", 
        'values': [True, False],
        'defaultValue': False
    }
}


clustering_indices_params = {
    
}

fasttext_input_boxes = [
    modelParamViewInputBox(**fasttext_params[i]) 
        for i in ['lr', 'dim', 'ws', 'minCount', 'minn', 'maxn', 'neg', 'wordNgrams']
]

fasttext_dropdowns = [  
    modelParamViewDropDown(**fasttext_params[i]) 
        for i in ['loss', 'model']
]

glove_input_boxes = [
    modelParamViewInputBox(**glove_params[i]) 
        for i in ['vocab_min_count', 'vector_size', 'max_iter', 'window_size', 'x_max']
]

bert_input_boxes = [
    modelParamViewInputBox(**bert_params['attention_probs_dropout_prob']),
]

roberta_gpt2_input_boxes = [
    modelParamViewInputBox(**roberta_gpt2_params[i])
        for i in ['num_train_epochs', 'learning_rate', 'train_batch_size', 'warmup_steps', 'max_seq_length']
]

word2vec_input_boxes = [
    modelParamViewInputBox(**word2vec_params[i]) for i in ['vector_size', 'window', 'min_count', 'alpha'] 
]

word2vec_dropdowns = [  
    modelParamViewDropDown(**word2vec_params[i]) 
        for i in ['hs']
]

doc2vec_input_boxes = [
    modelParamViewInputBox(**doc2vec_params[i]) for i in ['vector_size', 'window', ] 
]

sentence_bert_input_boxes = [
    modelParamViewInputBox(**sentence_bert_params[i]) 
        for i in ['batch_size']
]

sentence_bert_dropdowns = [
    modelParamViewDropDown(**sentence_bert_params[i])
        for i in ['normalize_embeddings', 'pretrained_model']
]

kmeans_dropdowns = [
    modelParamViewDropDown(**kmeans_params[i]) for i in ['init', 'algorithm'] 
]

kmeans_input_boxes = [
    modelParamViewInputBox(**kmeans_params[i]) for i in ['n_init', 'max_iterint'] 
]

agglomerative_dropdowns = [
    modelParamViewDropDown(**agglomerative_params[i]) for i in ['affinity', 'linkage'] 
]

clustering_indices_dropdowns= [
    
]

