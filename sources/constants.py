import os

available_methods = {
    "word2vec": ["avg", "tf-idf"],
    # "glove": ["avg", "tf-idf", 'pre-prepared-embeddings'], 
    # "fasttext": ['avg', 'tf-idf'], 
    # "bert": ["sentence-avg"], 
    # "roberta": ["sentence-avg", "sentence-avg2"], 
    # "gpt2": ["sentence-avg", "sentence-avg2"],
    # "scibert": ["sentence-avg", "sentence-avg2"],
    # "skip thoughts": ["sentence-avg"],
    # "doc2vec": ["doc-embedding"],
    # "sentence bert": ["sentence-avg"]
}

method_option_desc = {
    'avg': 'Averaging Words',
    'tf-idf': 'Tf-Idf Weighing Words',
    'sentence-avg': 'Averaging Sentences',
    'pre-prepared-embeddings': 'Using Pre-prepared Embeddings',
    'doc-embedding': 'Document Embedding',
    'sentence-avg2': 'Averaging Sentences(All Docs Trained Once)'
}

scores_to_params = {
    'hierarthical_scores': 'hierarthical_params',
    'kmeans_scores': 'kmeans_params',
    'dbscan_scores': 'dbscan_params'
}

index_value_key = {
    'Silhouette Index': 'silhouette_score',
    'DB Index': 'db_score',
    'Calinski Harabasz Score': 'calinski_harabasz_score',
    'Rand Index': 'rand_index',
    'Homogeneity Score': 'homogeneity_score'
}

initial_files_path=os.path.abspath('../initial-files')
connection_string='mongodb://127.0.0.1:27017'
dbname='cappadocia'