import numpy as np
import os, re
from subprocess import Popen

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO

from nltk.tokenize import word_tokenize

GLOVE_FOLDER = 'glove/'
SAVE_FILE=GLOVE_FOLDER + 'vectors_glove.txt'
GLOVE_SHELL_SCRIPT_FILE = GLOVE_FOLDER + 'ours.sh'

class GlovePipeline(Pipeline):
	def __init__(self, search_space = [], int_params = []):
		super(GlovePipeline, self).__init__(TrialsDAO("glove", 'avg'))
		self.search_space = search_space
		self.int_params = int_params
		self.data_list = [] # according to data format
		self.paper_names = []

	def train(self, params):
		def make_params_upper(params):
			params_upper = {}
			for key in params:
				params_upper[key.upper() + "="] = str(params[key]) 
			return params_upper
		def change_shell_file(params):
		    fd = open(GLOVE_SHELL_SCRIPT_FILE, "r")
		    file_content = ""
		    params_upper = make_params_upper(params)
		    for line in fd:
		        for key in params_upper:
		            if(key in line):
		                line = key + params_upper[key] + "\n"
		                break
		        file_content += line
		    with open(GLOVE_SHELL_SCRIPT_FILE, "w") as fd:
		        fd.write(file_content)
		def realize_train():
			wd = os.getcwd()
			os.chdir(GLOVE_FOLDER)
			p = Popen(["sh", "ours.sh"]) # opening given process
			p.wait()
			os.chdir(wd)
		def load_model():
			glove_file = SAVE_FILE
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
			for filename in self.filepaths:
				with open(filename) as fd:
					data = fd.read()
					paper_name = data.split('\n')[0]
					# data_list = re.split('\s+|\.', data)
					data_list = word_tokenize(data.lower())
					# data_list = list(filter(lambda a: a != '', data_list))
					if(len(data_list) > 0):
						doc_embed_avr = np.zeros(dim)
						counter = 0
						for word in data_list:
						    # word = re.sub(r'\W+', '', word).lower();
						    if(word != '' and word in model):
						        doc_embed_avr = np.add(model[word], doc_embed_avr)
						        counter += 1
						doc_embed_avr = np.divide(doc_embed_avr, counter)
						paper_embedding_dict[paper_name] = list(doc_embed_avr)
			return paper_embedding_dict
		# run starts here
		change_shell_file(params)
		realize_train()
		return create_vectors(load_model(), params['vector_size'])