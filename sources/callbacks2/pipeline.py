import os
from os import walk
from hyperopt import fmin, tpe
from statistics import mean as avg 
from sklearn.cluster import AgglomerativeClustering, KMeans
from callbacks2.ScoreCalculator import ScoreCalculator
from hyperopt import STATUS_FAIL
from hyperopt.fmin import generate_trial

from constants import initial_files_path
import time

class Pipeline:
	def __init__(self, trialsDAO=None):
		self.embeddings_dict = {}
		self.int_params = []
		self.init_filepaths()
		self.trialsDAO = trialsDAO

	def init_filepaths(self):
		path = initial_files_path + '/documents'
		if os.path.isdir(path): 
			pwd = os.getcwd()
			os.chdir(path)
			self.filepaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.txt'];
			os.chdir(pwd)
		else:
			self.filepaths = None

	def recreate_params(self, params):
		params = params[0]
		for key in self.int_params:
			if(key in self.int_params):
				params[key] = int(params[key])
		return params

	def add_params_to_trials(self, trials, params):
		print("add_params_to_trials")
		def get_new_tid(trials):
			tids = list(map(lambda y: y['misc']['tid'], trials.trials))
			return max(tids) + 1
		new_tid = get_new_tid(trials)
		new_trial = [generate_trial(new_tid, params)]
		print("trial object")
		trials.insert_trial_docs(new_trial)
		print("trial object inserted")
		return trials

	def run(self, manuel_params = None):
		(_id, trials) = self.trialsDAO.getTrialsFromDb()
		if(manuel_params != None):
			trials = self.add_params_to_trials(trials, manuel_params)
		best = fmin(fn=self.update_output_obj_fun,
		    space=self.search_space,
		    algo=tpe.suggest,
		    max_evals=len(trials) + 1,
		    trials=trials,
		)
		self.trialsDAO.saveTrialsToDb(trials,  _id)

	def update_output_obj_fun(self, params):
		try:
			tic = time.perf_counter()
			params = self.recreate_params(params)
			print("params: ", params)
			doc_embeddings_dict = self.train(params)
			score_calculator = ScoreCalculator()
			scores = score_calculator.calculate_scores(doc_embeddings_dict)
			silhouette_scores_lst = scores['optimized_scores']['hierarthical_scores']['silhouette_score']
			toc = time.perf_counter()
			doc_embeddings_dict = dict(doc_embeddings_dict)
			if(len(doc_embeddings_dict) > 1000):
				print("document embeddings size is very big. Time to use GridFS")
				document = dict({'time(sec.)': toc-tic, 'scores': scores['scores'], 'optimized_scores': scores['optimized_scores'], 'avg_score': avg(silhouette_scores_lst)}, **params)
				self.trialsDAO.save_big_document(document, doc_embeddings_dict)
			else:
				document = dict({'time(sec.)': toc-tic, 'scores': scores['scores'], 'optimized_scores': scores['optimized_scores'], 'avg_score': avg(silhouette_scores_lst), 'embeddings': doc_embeddings_dict}, **params)
				self.trialsDAO.save_document(document)
			return -avg(silhouette_scores_lst)
		except Exception as e:
			print(e)
			return {'status':STATUS_FAIL}


 