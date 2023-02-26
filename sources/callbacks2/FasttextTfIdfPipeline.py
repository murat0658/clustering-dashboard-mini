import fasttext
from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO
import numpy as np
import re
import pickle

from constants import initial_files_path


class FasttextTfIdfPipeline(Pipeline):
	def __init__(self, search_space = [], int_params = []):
		super(FasttextTfIdfPipeline, self).__init__(TrialsDAO("fasttext", 'tf-idf'))
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
		def read_tf_idf_fasttext():
			with open(initial_files_path + '/tf_idf_fasttext', 'rb') as handle:
				tf_idf_fasttext = pickle.load(handle)
			tf_idf_fasttext_dict = tf_idf_fasttext['tf_idf_fasttext_dict']
			tf_idf_fasttext_paper_total_dict = tf_idf_fasttext['tf_idf_fasttext_paper_total_dict']
			return (tf_idf_fasttext_dict, tf_idf_fasttext_paper_total_dict)

		def create_vectors(model, dim):
			(tf_idf_fasttext_dict, tf_idf_fasttext_paper_total_dict) = read_tf_idf_fasttext()
			paper_embedding_dict = {}
			for filename in self.filepaths:
				with open(filename, 'r') as fd:                    
					data = fd.read()
				paper_name = data.split('\n')[0]
				data_list = data.split()
				doc_embed_avr = np.zeros(dim)
				for word in data_list:
					if(word in model):
						embedding = model[word] * tf_idf_fasttext_dict[paper_name][word]
						doc_embed_avr = np.add(doc_embed_avr, embedding)
				denominator = (len(data_list) * tf_idf_fasttext_paper_total_dict[paper_name])
				doc_embed_avr = np.divide(doc_embed_avr, denominator)
				if(len(data_list) > 0):
					paper_embedding_dict[paper_name] = list(doc_embed_avr)
			return paper_embedding_dict
		word_embeddings = fasttext.train_unsupervised(initial_files_path + '/fullData.txt', **params)
		return create_vectors(word_embeddings, params['dim'])
