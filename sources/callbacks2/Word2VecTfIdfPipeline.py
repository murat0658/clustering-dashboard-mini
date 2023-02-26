from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
import pickle

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO
from constants import initial_files_path


class Word2VecTfIdfPipeline(Pipeline):
	def __init__(self, search_space = [], int_params = []):
		super(Word2VecTfIdfPipeline, self).__init__(TrialsDAO("word2vec", 'tf-idf'))
		self.search_space = search_space
		self.int_params = int_params
		with open(initial_files_path + '/sentences.txt') as fd:
			sentences = fd.read()
		sentences_lst = sentences.split('\n')
		self.tokens_of_sentences = [word_tokenize(sentence) for sentence in sentences_lst]
		with open(initial_files_path + '/tf_idf_glove', 'rb') as handle:
			tf_idf_glove = pickle.load(handle)
		self.tf_idf_glove_dict = tf_idf_glove['tf_idf_glove_dict']
		self.tf_idf_glove_paper_total_dict = tf_idf_glove['tf_idf_glove_paper_total_dict']

	def train(self, params):
		print("model is started")
		model = Word2Vec(self.tokens_of_sentences, **params)
		print("model is created")
		paper_embedding_dict = {}
		for filepath in self.filepaths:
			with open(filepath, 'r') as fd:
				data  = fd.read()
			paper_name = data.split('\n')[0]
			doc_tokens = word_tokenize(data)
			doc_embed_avr = np.zeros(params['vector_size'])
			counter = 0
			for token in doc_tokens:
				if(token in model.wv and token in self.tf_idf_glove_dict[paper_name]):
					embedding = (model.wv[token] * self.tf_idf_glove_dict[paper_name][token])
					doc_embed_avr = np.add(embedding, doc_embed_avr)
					counter += 1
			denominator = (counter * self.tf_idf_glove_paper_total_dict[paper_name])
			print(counter)
			if(counter  == 0):
				paper_embedding_dict[paper_name] = 0
			else:
				doc_embed_avr = np.divide(doc_embed_avr, denominator)
				paper_embedding_dict[paper_name] = list(doc_embed_avr)
		return paper_embedding_dict

