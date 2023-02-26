from Doc2vecPipeline import Doc2vecPipeline
from hyperopt import hp
path = "/Users/muratkara/Desktop/yüksek lisans çalışmalar/phase13/real/refactored_papers"
doc2vec_search_space = [{
    'vector_size': hp.quniform('vector_size', 10, 250, 5),
    'window': hp.quniform('window', 1, 20, 1)
}]

Doc2vecPipeline(path, doc2vec_search_space, ['vector_size', 'window']).run("doc2vec", "avg")