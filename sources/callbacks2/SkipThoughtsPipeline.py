# -*- coding: utf-8 -*-
import numpy as np
import os, re
from subprocess import Popen
import pickle

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO

from constants import initial_files_path

class SkipThoughtsPipeline(Pipeline):
	def __init__(self, search_space = [{}], int_params = []):
		super(SkipThoughtsPipeline, self).__init__(TrialsDAO("skip thoughts", 'sentence-avg'))
		self.search_space = search_space
		self.int_params = int_params
		self.paper_names = []
		self.init_data()

	def init_data(self):
		for filename in self.filepaths:
			with open(filename, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			self.paper_names.append(paper_name)

	def train(self, params):
		def load_embeddings_lst():
			print("training skip thoughts vectors")
			wd = os.getcwd()
			os.chdir("skip-thoughts")
			command = "python run.py"
			command_lst = command.split()
			p = Popen(command_lst)
			p.wait()
			print("skip thoughts embeddings are trained")
			os.chdir(wd)
			with open(initial_files_path +  '/skipthoughts.pickle', 'rb') as handle:
				return pickle.load(handle, encoding='latin1')
		def give_titles_to_vectors(all_embeddings_lst):
			doc_embeddings_dict = {}
			for (i, title) in enumerate(self.paper_names):
				doc_embeddings_dict[title] = [float(j)for j in all_embeddings_lst[i]][:2400]
			return doc_embeddings_dict
		doc_embeddings_list = load_embeddings_lst()
		return give_titles_to_vectors(doc_embeddings_list)