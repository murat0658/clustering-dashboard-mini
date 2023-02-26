import fasttext
from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO
import numpy as np
from nltk.tokenize import word_tokenize

from constants import initial_files_path

class FasttextPipeline(Pipeline):
	def __init__(self, search_space = [], int_params = []):
		super(FasttextPipeline, self).__init__(TrialsDAO("fasttext", 'avg'))
		self.search_space = search_space
		self.int_params = int_params

	def train(self, params):
		def create_vectors(model, dim):
			paper_embedding_dict = {}
			c = 0
			for filename in self.filepaths:
				if(c%1000 == 0):
					print("Document Number: ", c)
				c += 1
				with open(filename, 'r') as fd:
					data = fd.read()
				paper_name = data.split('\n')[0]
				data_list = word_tokenize(data)
				doc_embed_avr = np.zeros(dim)
				for word in data_list:
					if(word in model):
						doc_embed_avr = np.add(doc_embed_avr, model[word])
				doc_embed_avr = np.divide(doc_embed_avr, len(data_list))
				if(len(data_list) > 0):
					paper_embedding_dict[paper_name] = list(doc_embed_avr)
			return paper_embedding_dict

		word_embeddings = fasttext.train_unsupervised(initial_files_path  + "/fullData.txt", **params)
		return create_vectors(word_embeddings, params['dim'])
