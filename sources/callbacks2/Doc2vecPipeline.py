from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Doc2vecPipeline(Pipeline):
    def __init__(self, search_space = [], int_params = []):
        super(Doc2vecPipeline, self).__init__(TrialsDAO("doc2vec", 'doc-embedding'))
        self.search_space = search_space
        self.int_params = int_params
        self.data_list = [] # according to data format
        self.paper_names = []
        self.init_data()

    def init_data(self):
        for filename in self.filepaths:
            with open(filename, 'r') as fd:
                data = fd.read()
            paper_name = data.split('\n')[0]
            self.paper_names.append(paper_name)
            data_list = data.split()
            data_list = list(filter(lambda a: a != '', data_list))
            self.data_list.append(data_list)

    def train(self, params):
        def create_tagged_document():
            return [TaggedDocument(doc, [i]) for i, doc in enumerate(self.data_list)]
        documents = create_tagged_document()
        model = Doc2Vec(documents, **params)
        embeddings_dict = {}
        for i, data_list in enumerate(self.data_list):
            vector = model.infer_vector(data_list)
            embeddings_dict[self.paper_names[i]] = list([float(i) for i in vector])

        return embeddings_dict