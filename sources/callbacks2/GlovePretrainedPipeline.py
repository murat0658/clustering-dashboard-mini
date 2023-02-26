import numpy as np
import re

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO

from constants import initial_files_path

class GlovePretrainedPipeline(Pipeline):
	def __init__(self, search_space = [], int_params = []):
		super(GlovePretrainedPipeline, self).__init__(TrialsDAO("glove", 'pre-prepared-embeddings'))
		self.search_space = search_space
		self.int_params = int_params
		self.data_list = [] # according to data format
		self.paper_names = []

	def train(self, params):
		def load_model(pretrained_file):
			glove_file = initial_files_path + '/../glove_models/' + pretrained_file
			gloveModel = {}
			with open(glove_file,'r') as f:
				for line in f:
				    splitLines = line.split()
				    word = splitLines[0]
				    wordEmbedding = np.array([float(value) for value in splitLines[1:]])
				    gloveModel[word] = wordEmbedding
			print(len(gloveModel)," words loaded!")
			return gloveModel
		def create_vectors(model, dim):
			paper_embedding_dict = {}
			c = 0
			for filename in self.filepaths:
				with open(filename) as fd:
					data = fd.read()
					paper_name = data.split('\n')[0]
					data_list = re.split('\s+|\.', data)
					data_list = list(filter(lambda a: a != '', data_list))
					if(len(data_list) > 0):
						doc_embed_avr = np.zeros(dim)
						for word in data_list:
						    word = re.sub(r'\W+', '', word).lower();
						    if(word != '' and word in model):
						        doc_embed_avr = np.add(model[word], doc_embed_avr)
						        c+=1
						doc_embed_avr = np.divide(doc_embed_avr, len(data_list))
						paper_embedding_dict[paper_name] = list(doc_embed_avr)
			print("counter: ", c)
			return paper_embedding_dict
		# run starts here
		pretrained_file_name = params['pretrained_model']
		print(pretrained_file_name)
		vector_size = int(pretrained_file_name.split('.')[2][:-1])
		return create_vectors(load_model(pretrained_file_name), vector_size)