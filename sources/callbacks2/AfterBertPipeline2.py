import numpy as np
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO
from constants import initial_files_path

class AfterBertPipeline2(Pipeline):
	def __init__(self, method, search_space = [{}], int_params = []):
		super(AfterBertPipeline2, self).__init__(TrialsDAO(method[2], 'sentence-avg2'))
		self.search_space = search_space
		self.int_params = int_params
		self.paper_names = []
		self.sentences_path = initial_files_path + '/sentences.txt'
		self.method = method
		self.init_data()

	def init_data(self):
		print("init data started")
		for filename in self.filepaths:
			with open(filename, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			self.paper_names.append(paper_name)
			with open(self.sentences_path, "r") as fd:
				data = fd.read()
			sentences = data.split('\n')
			indices = [i for i, x in enumerate(sentences) if x == '']
			indices = [indices[i] for i in range(0,len(indices), 2)]
			self.indices = []
			c = 0
			# dökümanlar arası iki satır boşluk var. bu yüzden ilki 0 olmak üzere
			# index'ler 2, 4, 6, .. şeklinde sola kayıyor
			for index in indices:
				self.indices.append(index-c)
				c +=2
			self.sentences_lst = list(filter(lambda x:x!='',sentences))
		print("init data ended")

	def train(self, params):
		def create_embeddings_lst(model):
			print("train started")
			all_sentence_embeddings = model.encode_sentences(self.sentences_lst, combine_strategy="mean")
			print("train ended")
			prev_index = 0
			all_embeddings_lst = []
			for index in self.indices:
				all_embeddings_lst.append(list(np.mean(all_sentence_embeddings[prev_index:index], axis=0)))
				prev_index = index
			print("recreating data ended")
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