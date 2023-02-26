import numpy as np
from sentence_transformers import SentenceTransformer
import os, re

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO

from constants import initial_files_path


class SentenceBertPipeline(Pipeline):
	def __init__(self, search_space = [], int_params = [], ):
		super(SentenceBertPipeline, self).__init__(TrialsDAO("sentence bert", 'sentence-avg'))
		self.search_space = search_space
		self.int_params = int_params
		self.paper_names = []
		self.sentences_path = initial_files_path + '/sentences.txt'
		self.init_data()

	def init_data(self):
		for filename in self.filepaths:
			with open(filename, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			self.paper_names.append(paper_name)

	def train(self, params):
		def create_embeddings_lst(model, params):
			all_embeddings_lst = []
			doc_sentences = []
			with open(self.sentences_path, "r") as fd:
				empty_line = False
				for line in fd:
					if(line == "\n"):
						if empty_line:
							doc_sentence_embeddings = model.encode(doc_sentences, show_progress_bar=True, **params)
							all_embeddings_lst.append(list(np.mean(doc_sentence_embeddings, axis=0)))
							empty_line = False
							doc_sentences = []
						else:
							empty_line = True
					else:
						doc_sentences.append(line)
			doc_sentence_embeddings = model.encode(doc_sentences, show_progress_bar=True, **params)
			all_embeddings_lst.append(list(np.mean(doc_sentence_embeddings, axis=0)))
			return all_embeddings_lst

		def give_titles_to_vectors(all_embeddings_lst):
			doc_embeddings_dict = {}
			for (i, title) in enumerate(self.paper_names):
				doc_embeddings_dict[title] = list(all_embeddings_lst[i])
			return doc_embeddings_dict

		def realize_train(params):
			model = SentenceTransformer(params['pretrained_model'])
			print("params: " , params)
			new_params = {}
			for key in params:
				if key != 'pretrained_model':
					new_params[key] = params[key]
			print("new_params: " , new_params)
			return create_embeddings_lst(model, new_params)

		all_embeddings_lst = realize_train(params)
		all_embeddings_lst = [[float(j) for j in i] for i in all_embeddings_lst]
		return give_titles_to_vectors(all_embeddings_lst)
