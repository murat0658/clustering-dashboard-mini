from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO
from constants import initial_files_path


class Word2VecPipeline(Pipeline):
	def __init__(self, search_space = [], int_params = []):
		super(Word2VecPipeline, self).__init__(TrialsDAO("word2vec", 'avg'))
		self.search_space = search_space
		self.int_params = int_params
		with open(initial_files_path + '/sentences.txt') as fd:
			sentences = fd.read()
		sentences_lst = sentences.split('\n')
		self.tokens_of_sentences = [word_tokenize(sentence) for sentence in sentences_lst]

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
				if(token in model.wv):
					doc_embed_avr = np.add(model.wv[token], doc_embed_avr)
					counter += 1
			if(counter  == 0):
				paper_embedding_dict[paper_name] = 0
			else:
				doc_embed_avr = np.divide(doc_embed_avr, counter)
				paper_embedding_dict[paper_name] = list(doc_embed_avr)
		return paper_embedding_dict

