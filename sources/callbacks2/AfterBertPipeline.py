import numpy as np
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO
from constants import initial_files_path

class AfterBertPipeline(Pipeline):
	def __init__(self, method, search_space = [{}], int_params = []):
		super(AfterBertPipeline, self).__init__(TrialsDAO(method[2], 'sentence-avg'))
		self.search_space = search_space
		self.int_params = int_params
		self.paper_names = []
		self.sentences_path = initial_files_path + '/sentences.txt'
		self.method = method
		self.init_data()

	def init_data(self):
		for filename in self.filepaths:
			with open(filename, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			self.paper_names.append(paper_name)

	def train(self, params):
		def create_embeddings_lst(model):
			all_embeddings_lst = []
			doc_sentences = []
			c = 0.0
			with open(self.sentences_path, "r") as fd:
				empty_line = False
				for line in fd:
					if(line == "\n"):
						if empty_line:
							doc_sentence_embeddings = model.encode_sentences(doc_sentences, combine_strategy="mean")
							all_embeddings_lst.append(list(np.mean(doc_sentence_embeddings, axis=0)))
							empty_line = False
							doc_sentences = []
							c+=1.0
							print(c/600.0)
						else:
							empty_line = True
					else:
						doc_sentences.append(line)
			doc_sentence_embeddings = model.encode_sentences(doc_sentences, combine_strategy="mean")
			all_embeddings_lst.append(list(np.mean(doc_sentence_embeddings, axis=0)))
			return all_embeddings_lst

		def give_titles_to_vectors(all_embeddings_lst):
			doc_embeddings_dict = {}
			for (i, title) in enumerate(self.paper_names):
				doc_embeddings_dict[title] = all_embeddings_lst[i]
			return doc_embeddings_dict

		model_args = ModelArgs(**params)
		model = RepresentationModel(
            self.method[0],
            self.method[1],
            use_cuda = False,
            args=model_args,
        )
		all_embeddings_lst = create_embeddings_lst(model)
		all_embeddings_lst = [[float(j) for j in i] for i in all_embeddings_lst]
		return give_titles_to_vectors(all_embeddings_lst)