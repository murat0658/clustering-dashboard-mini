from genericpath import isfile
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, homogeneity_score
import os, json
from constants import available_methods, index_value_key
from callbacks2.TrialsDAO import TrialsDAO
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from itertools import chain
from hyperopt import Trials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
import pickle 
from statistics import mean

from constants import initial_files_path
from search_spaces import kmeans_search_space, kmeans_int_params, agglomerative_search_space, dbscan_search_space, dbscan_int_params

class MetricScoreCalculator():
	def __init__(self, n_clusters, search_space, clustering_method, int_params, embeddings):
		self.search_space = search_space
		self.clustering_method = clustering_method
		self.int_params = int_params
		self.n_clusters = n_clusters
		self.embeddings = embeddings

	def set_n_clusters(self, n_clusters):
		self.n_clusters = n_clusters
		return self

	def recreate_params(self, params):
		params = params[0]
		for key in self.int_params:
			if(key in self.int_params):
				params[key] = int(params[key])
		return params

	def obj_fun(self, params):
		raise NotImplementedError("Objective function must be implemented");
	
	def get_labels(self, params = {}):
		if(params != {}):
			params = self.recreate_params(params)
		clustering_method_instance = None
		if(self.n_clusters != None):
			clustering_method_instance = self.clustering_method(n_clusters=self.n_clusters, **params)
		else:
			clustering_method_instance = self.clustering_method(**params)
		return clustering_method_instance.fit(self.embeddings).labels_.tolist()

	def calculate_optimized_score_unlabeled(self):
		trials= Trials()
		try:
			fmin(fn=self.obj_fun,
			    space=self.search_space,
			    algo=tpe.suggest,
			    max_evals=10,
			    trials=trials,
			)
		except Exception as err:
			return None
		return trials.best_trial

class SilhouetteScoreCalculator(MetricScoreCalculator):
	def __init__(self, embeddings, clustering_method, search_space, int_params = [], n_clusters = None):
		super().__init__(n_clusters, search_space, clustering_method, int_params, embeddings)

	def obj_fun(self, params):
		labels = self.get_labels(params)
		if len(set(labels)) < 2:
			return {'status': STATUS_FAIL }
		score = silhouette_score(X=self.embeddings, labels=labels)
		return -score

	def calculate(self):
		best_trial = self.calculate_optimized_score_unlabeled()
		if(best_trial == None):
			return (None, None)
		return (-best_trial['result']['loss'], best_trial['misc']['vals'])

	def calculate_default(self):
		labels = self.get_labels()
		if len(set(labels)) < 2:
			return None
		return silhouette_score(X=self.embeddings, labels=labels)

class DBScoreCalculator(MetricScoreCalculator):
	def __init__(self, embeddings, clustering_method, search_space, int_params = [], n_clusters = None):
		super().__init__(n_clusters, search_space, clustering_method, int_params, embeddings)

	def obj_fun(self, params):
		labels = self.get_labels(params)
		if len(set(labels)) < 2:
			return {'status': STATUS_FAIL }
		score = davies_bouldin_score(X=self.embeddings, labels=labels)
		return score

	def calculate(self):
		best_trial = self.calculate_optimized_score_unlabeled()
		if(best_trial == None):
			return (None, None)
		return (best_trial['result']['loss'], best_trial['misc']['vals'])

	def calculate_default(self):
		labels = self.get_labels()
		if len(set(labels)) < 2:
			return None
		return davies_bouldin_score(X=self.embeddings, labels=labels)

class CalinskiHarabaszCalculator(MetricScoreCalculator):
	def __init__(self, embeddings, clustering_method, search_space, int_params, n_clusters = None):
		super().__init__(n_clusters, search_space, clustering_method, int_params, embeddings)

	def obj_fun(self, params):
		labels = self.get_labels(params)
		if len(set(labels)) < 2:
			return {'status': STATUS_FAIL }
		score = calinski_harabasz_score(X=self.embeddings, labels=labels)
		return -score

	def calculate(self):
		best_trial = self.calculate_optimized_score_unlabeled()
		if(best_trial == None):
			return (None, None)
		return (-best_trial['result']['loss'], best_trial['misc']['vals'])

	def calculate_default(self):
		labels = self.get_labels()
		if len(set(labels)) < 2:
			return None
		return calinski_harabasz_score(X=self.embeddings, labels=labels)


class LabeledMetricCalculator(MetricScoreCalculator):
	def __init__(self, embeddings, clustering_method, search_space=[], int_params=[]):
		self.annotated_docs_dict = self.read_annotation_file()
		self.init_filepaths()
		n_clusters = 0
		if self.annotated_docs_dict != None :
			n_clusters = len(self.annotated_docs_dict)
		super().__init__(n_clusters, search_space, clustering_method, int_params, embeddings)

	def init_filepaths(self):
		path = initial_files_path + '/documents'
		if os.path.isdir(path): 
			pwd = os.getcwd()
			os.chdir(path)
			self.filepaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.txt'];
			os.chdir(pwd)
		else:
			self.filepaths = None

	def get_true_labels_mixed(self):
		annotated_docs = []
		for lst in self.annotated_docs_dict.values():
			annotated_docs.append(lst)
		return [len(el) * [i] for i,el in enumerate(annotated_docs)]


	def get_predicted_labels(self):
		labelled_papers_indices2D = self.get_labelled_papers_indices2D()
		n_clusters = len(labelled_papers_indices2D)
		labels = self.get_labels()
		print("labels: ", labels)
		predicted_labels = [ [labels[labelled_paper_index] 
					for labelled_paper_index in labelled_papers_indices] 
				for labelled_papers_indices in labelled_papers_indices2D ];
		return predicted_labels

	def calculate_true_and_predicted_labels(self):
		true_labels = []
		predicted_labels = self.get_predicted_labels()
		n_clusters = len(predicted_labels)
		label_count_ratios = [[predicted_labels1D.count(i)/len(predicted_labels1D) 
										for i in range(n_clusters)] 
									for predicted_labels1D in predicted_labels]
		label_preference_lst = [[(t[0],t[1]) 
											for t in sorted(enumerate(label_count_ratios1D), key=lambda k: k[1], reverse=True)]
										for label_count_ratios1D in label_count_ratios]
		cluster_match_lst = []
		for k in range(n_clusters):
			current_cluster_lst = []
			j = 0;
			isFound = False;
			while isFound == False:
				for i in range(n_clusters):
					label = label_preference_lst[i][j][0]
					if(label == k and i not in cluster_match_lst):
						current_cluster_lst.append(i)
				if len(current_cluster_lst) == 0:
					j += 1
				elif len(current_cluster_lst) == 1:
					cluster_match_lst.append(current_cluster_lst[0])
					isFound = True
				elif len(current_cluster_lst) > 1:
					isFound = True
					max_ratio = -1;
					max_ratio_index = -1;
					for cluster_number in current_cluster_lst:
						if(label_preference_lst[cluster_number][j][1] > max_ratio):
							max_ratio = label_preference_lst[cluster_number][j][1]
							max_ratio_index = cluster_number
					cluster_match_lst.append(max_ratio_index)
		true_labels = [[]] * n_clusters
		for i, cluster_match in enumerate(cluster_match_lst):
			true_labels[cluster_match] = [i] * len(predicted_labels[cluster_match])
		return (list(chain.from_iterable(true_labels)), 
			list(chain.from_iterable(predicted_labels)))


	def get_labelled_papers_indices2D(self):
		annotated_docs = []
		for lst in self.annotated_docs_dict.values():
			annotated_docs.append(lst)

		paper_names = []
		for filepath in self.filepaths:
			with open(filepath, "r") as file:
				paper_names.append(file.readline().rstrip())
		res = []
		for cluster_lst in annotated_docs:
			res1D = []
			for annotated_doc in cluster_lst:
				res1D.append(paper_names.index(annotated_doc))
			res.append(res1D)
		return res

	def read_annotation_file(self):
		if os.path.isfile(initial_files_path + '/annotations.json'):
			with open(initial_files_path + '/annotations.json', 'r') as fd:
				data = fd.read()
			return json.loads(data)
		return None


class AdjustedRandScoreCalculator(LabeledMetricCalculator):
	def __init__(self, embeddings, clustering_method, search_space, int_params = []):
		super().__init__(embeddings, clustering_method, search_space, int_params)

	def obj_fun(self, params):
		(true_labels, predicted_labels) = self.calculate_true_and_predicted_labels()
		score = adjusted_rand_score(labels_true=true_labels, labels_pred=predicted_labels)
		return -score

	def calculate(self):
		best_trial = self.calculate_optimized_score_unlabeled()
		return (-best_trial['result']['loss'], best_trial['misc']['vals'])

	def calculate_default(self):
		(true_labels, predicted_labels) = self.calculate_true_and_predicted_labels()
		return adjusted_rand_score(labels_true=true_labels, labels_pred=predicted_labels)


class HomogeneityScoreCalculator(LabeledMetricCalculator):
	def __init__(self, embeddings, clustering_method, search_space, int_params = []):
		super().__init__(embeddings, clustering_method, search_space, int_params)

	def obj_fun(self, params):
		true_labels = list(chain.from_iterable(self.get_true_labels_mixed()))
		predicted_labels = list(chain.from_iterable(self.get_predicted_labels()))
		score = homogeneity_score(labels_true=true_labels, labels_pred=predicted_labels)
		return -score

	def calculate(self):
		best_trial = self.calculate_optimized_score_unlabeled()
		return (-best_trial['result']['loss'], best_trial['misc']['vals'])

	def calculate_default(self):
		true_labels = list(chain.from_iterable(self.get_true_labels_mixed()))
		predicted_labels = list(chain.from_iterable(self.get_predicted_labels()))
		return homogeneity_score(labels_true=true_labels, labels_pred=predicted_labels)
		

class DBSCANScoreCalculator():
	def calculate_optimized_scores(self, embeddings):
		search_space = dbscan_search_space
		int_params = dbscan_int_params
		_silhouette_val = SilhouetteScoreCalculator(embeddings, DBSCAN, search_space, int_params).calculate()
		_db_val = DBScoreCalculator(embeddings, DBSCAN, search_space, int_params).calculate()
		_calisnki_harabasz_val = CalinskiHarabaszCalculator(embeddings, DBSCAN, search_space, int_params).calculate()
		return ({
				    'silhouette_score': _silhouette_val[0],
				    'db_score': _db_val[0],
				    'calinski_harabasz_score': _calisnki_harabasz_val[0]
				},{
				    'silhouette_score_param': _silhouette_val[1],
				    'db_score_param': _db_val[1],
				    'calinski_harabasz_score_param': _calisnki_harabasz_val[1]
				})

	def calculate_scores(self, embeddings):
		print("Calculating scores with DBSCAN")
		try:
			labels = DBSCAN().fit(embeddings).labels_.tolist()
		except Exception as err:
			return {
			    'silhouette_score': None,
			    'db_score': None,
			    # 'calinski_harabasz_score': None
			}
		if(len(set(labels)) < 2):
			return {
			    'silhouette_score': None,
			    'db_score': None,
			    'calinski_harabasz_score': None
			}
		return {
			    'silhouette_score': silhouette_score(X = embeddings, labels=labels),
			    'db_score': davies_bouldin_score(X = embeddings, labels=labels),
			    'calinski_harabasz_score': calinski_harabasz_score(X=embeddings, labels=labels)
			}

class ClusteringMethodCalculator():
	def __init__(self, clustering_method, search_space, int_params, embeddings):
		self.cluster_counts = range(2,21)
		self.silhouetteScoreCalculator = SilhouetteScoreCalculator(embeddings, clustering_method, search_space, int_params)
		self.dbScoreCalculator = DBScoreCalculator(embeddings, clustering_method, search_space, int_params)
		self.calinskiHarabaszCalculator = CalinskiHarabaszCalculator(embeddings, clustering_method, search_space, int_params)
		self.adjustedRandScoreCalculator = AdjustedRandScoreCalculator(embeddings, clustering_method, search_space, int_params)
		self.homogeneityScoreCalculator = HomogeneityScoreCalculator(embeddings, clustering_method, search_space, int_params)

	def calculate_optimized_scores(self):
		scores_dict = {} 
		scores = [[],[],[]]
		result_params = [[],[],[]]
		for cluster_count in self.cluster_counts:
			for i,calculator in enumerate([self.silhouetteScoreCalculator, self.dbScoreCalculator]):
				(score, param) = calculator.set_n_clusters(cluster_count).calculate()
				scores[i].append(score)
				result_params[i].append(param)
		if (os.path.isfile(initial_files_path + '/annotations.json')):
			(rand_score, rand_param) = self.adjustedRandScoreCalculator.calculate()
			(homogeneity_score, homogeneity_param) = self.homogeneityScoreCalculator.calculate()
			return  ({
							    'silhouette_score': scores[0],
							    'db_score': scores[1],
							    'calinski_harabasz_score': scores[2],
							    'rand_index': rand_score,
							    'homogeneity_score': homogeneity_score
							},{
								'silhouette_score_param': result_params[0],
							    'db_score_param': result_params[1],
							    'calinski_harabasz_score_param': result_params[2],
							    'rand_index_param': rand_param,""
							    'homogeneity_score_param': homogeneity_param
							})
		else:
			return ({
					    'silhouette_score': scores[0],
					    'db_score': scores[1],
					    'calinski_harabasz_score': scores[2]
					},{
					    'silhouette_score_param': result_params[0],
					    'db_score_param': result_params[1],
					    'calinski_harabasz_score_param': result_params[2]
					})

	def calculate_scores(self):
		scores_dict = {} 
		scores = [[],[],[],[],[]]
		for cluster_count in self.cluster_counts:
			for i,calculator in enumerate([self.silhouetteScoreCalculator, self.dbScoreCalculator, self.calinskiHarabaszCalculator]):
				scores[i].append(calculator.set_n_clusters(cluster_count).calculate_default())
		if (os.path.isfile(initial_files_path + '/annotations.json')):
			scores[3] = self.adjustedRandScoreCalculator.calculate_default()
			scores[4] = self.homogeneityScoreCalculator.calculate_default()
			return  {
				    'silhouette_score': scores[0],
				    'db_score': scores[1],
				    'calinski_harabasz_score': scores[2],
				    'rand_index': scores[3],
				    'homogeneity_score': scores[4]
				}
		else:
			return {
			    'silhouette_score': scores[0],
			    'db_score': scores[1],
			    'calinski_harabasz_score': scores[2]
			} 


class AgglomerativeScoreCalculator(ClusteringMethodCalculator):
	print("Calculating scores with Agglomerative")
	def __init__(self, embeddings):
		super().__init__(AgglomerativeClustering, agglomerative_search_space, [], embeddings)

class KMeansScoreCalculator(ClusteringMethodCalculator):
	print("Calculating scores with K-Means")
	def __init__(self, embeddings):
		super().__init__(KMeans, kmeans_search_space, kmeans_int_params, embeddings)

class ScoreCalculator:
	def get_annotation_cluster_number(self):
		if(os.path.isfile(initial_files_path + '/annotations.json')):
			with open(initial_files_path + '/annotations.json', 'r') as fd:
				data = fd.read()
			return len(json.loads(data))
		return -1

	
	def recalculate_scores(self):
		for embedding_method in available_methods:
			trialsDAO = TrialsDAO(embedding_method)
			documents = trialsDAO.get_all_docs()
			print("total number of documents: ", len(documents))
			for (c, document) in enumerate(documents):
				embeddings = list(document['embeddings'].values())
				# non-optimized results
				kmeans_score_calculator = KMeansScoreCalculator(embeddings)
				agglomerative_score_calculator = AgglomerativeScoreCalculator(embeddings)
				dbscan_calculator = DBSCANScoreCalculator()
				if 'scores' not in document:
					print(embedding_method)
					kmeans_scores = kmeans_score_calculator.calculate_scores()
					agglomerative_new_scores = agglomerative_score_calculator.calculate_scores()
					dbscan_new_scores = dbscan_calculator.calculate_scores(embeddings)

					trialsDAO.update_one_doc({'_id': document['_id']}, {'$set': {'scores': { 'kmeans_scores': kmeans_scores, 
					 					'hierarthical_scores': agglomerative_new_scores, 'dbscan_scores': dbscan_new_scores}}})

				if 'optimized_scores' not in document:
					print(embedding_method)
					# optimized results
					(kmeans_scores, kmeans_params) = kmeans_score_calculator.calculate_optimized_scores()
					(agglomerative_new_scores, agglomerative_params) = agglomerative_score_calculator.calculate_optimized_scores()
					(dbscan_new_scores, dbscan_params) = dbscan_calculator.calculate_optimized_scores(embeddings)

					trialsDAO.update_one_doc({'_id': document['_id']}, {'$set': {'optimized_scores': { 'kmeans_scores': kmeans_scores, 
										'hierarthical_scores': agglomerative_new_scores, 'dbscan_scores': dbscan_new_scores,
										'kmeans_params': kmeans_params, 'hierarthical_params': agglomerative_params, 
										'dbscan_params': dbscan_params }}})


				print("document number: ", c, "is calculated")


	
	def calculate_scores_with_lst(self, embeddings):
		kmeans_score_calculator = KMeansScoreCalculator(embeddings)
		agglomerative_score_calculator = AgglomerativeScoreCalculator(embeddings)
		dbscan_calculator = DBSCANScoreCalculator()
		d = {}
		kmeans_scores = kmeans_score_calculator.calculate_scores()
		agglomerative_new_scores = agglomerative_score_calculator.calculate_scores()
		# dbscan_new_scores = dbscan_calculator.calculate_scores(embeddings)
		# d['scores'] = { 'kmeans_scores': kmeans_scores, 
	 # 					'hierarthical_scores': agglomerative_new_scores, 
	 # 					'dbscan_scores': dbscan_new_scores}
		d['scores'] = { 'kmeans_scores': kmeans_scores, 
							'hierarthical_scores': agglomerative_new_scores }

		# optimized results
		(kmeans_scores, kmeans_params) = kmeans_score_calculator.calculate_optimized_scores()
		(agglomerative_new_scores, agglomerative_params) = agglomerative_score_calculator.calculate_optimized_scores()
		# (dbscan_new_scores, dbscan_params) = dbscan_calculator.calculate_optimized_scores(embeddings)
		# d['optimized_scores'] = { 'kmeans_scores': kmeans_scores, 'hierarthical_scores': agglomerative_new_scores, 'dbscan_scores': dbscan_new_scores,
		# 					'kmeans_params': kmeans_params, 'hierarthical_params': agglomerative_params, 
		# 					'dbscan_params': dbscan_params }
		d['optimized_scores'] = { 'kmeans_scores': kmeans_scores, 'hierarthical_scores': agglomerative_new_scores,
					'kmeans_params': kmeans_params, 'hierarthical_params': agglomerative_params, }


		return d		

	def calculate_scores(self, embeddings_dict):
		embeddings = list(embeddings_dict.values())
		print("hesaplama baslar")
		# non-optimized results
		return self.calculate_scores_with_lst(embeddings)

	def get_optimized_scores(self, embedding_method, embedding_method_option, index, clustering_method):
		one_element_list = TrialsDAO(embedding_method, embedding_method_option).get_optimized_params(index, clustering_method)
		print("one_element_list", one_element_list)
		if(one_element_list != []):
			return one_element_list[0]['optimized_params']

	def get_dbscan_cluster_number(self, embedding_method, embedding_method_option, clustering_index):
		clustering_index_key = index_value_key[clustering_index]
		one_element_list = TrialsDAO(embedding_method, embedding_method_option).get_document(clustering_index_key, 'dbscan_scores')
		if(one_element_list != []):
			embeddings = list(one_element_list[0]['embeddings'].values())
			labels = DBSCAN().fit(embeddings).labels_.tolist()
			return str(len(set(labels)))
		return "Not attendant"

	def get_scores_for_graph(self, index = 'Rand Index', clustering_method ='hierarthical_scores'):
		clustering_index_key = index_value_key[index]
		result = []
		scores_dict = {}
		for embedding_method in available_methods:
			for embedding_method_option in available_methods[embedding_method]:
				one_element_list = TrialsDAO(embedding_method, embedding_method_option).get_document(clustering_index_key, clustering_method)
				if(one_element_list != []):
					scores = one_element_list[0]['optimized_scores']
					embedding_key = str(embedding_method) + '-' + str(embedding_method_option)
					if(index == 'Rand Index' or index == 'Homogeneity Score'):
						cluster_count = self.get_annotation_cluster_number()
						result.append({'embedding method': embedding_key, 'cluster counts': cluster_count, 'scores': scores})
					elif clustering_method == 'dbscan_scores':
						if(index != 'Rand Index' and index != 'Homogeneity Score'):
							embeddings = list(one_element_list[0]['embeddings'].values())
							labels = DBSCAN().fit(embeddings).labels_.tolist()
							_len = len(set(labels))
							if _len <= 1:
								scores = None
							result.append({'embedding method': embedding_key, 'cluster counts': _len, 'scores': scores})
					else:
						scores_dict[embedding_key] = scores
						_range = list(range(2,21))
						for i, cc in enumerate(_range):
							result.append({'embedding method': embedding_key, 'cluster counts': cc, 'scores': scores[i]})
		return (result, scores_dict)

	def get_scores(self, index = 'Rand Index', clustering_method ='hierarthical_scores'):
		clustering_index_key = index_value_key[index]
		result = []
		print("available_methods: ", available_methods)
		for embedding_method in available_methods:
			for embedding_method_option in available_methods[embedding_method]:
				print("embedding_method_option: ", embedding_method_option, "embedding_method: ", embedding_method)
				one_element_list = TrialsDAO(embedding_method, embedding_method_option).get_document(clustering_index_key, clustering_method)
				if(one_element_list != []):
					scores = one_element_list[0]['optimized_scores']
					if(index == 'Rand Index' or index == 'Homogeneity Score'):
						cluster_count = self.get_annotation_cluster_number()
						print(embedding_method, embedding_method_option, scores)
						result +=  [{'method': embedding_method, 'option': embedding_method_option, str(cluster_count): scores}]
					elif clustering_method == 'dbscan_scores':
						if(index != 'Rand Index' and index != 'Homogeneity Score'):
							embeddings = list(one_element_list[0]['embeddings'].values())
							labels = DBSCAN().fit(embeddings).labels_.tolist()
							_len = len(set(labels))
							if _len <= 1:
								scores = None
							result +=  [{'method': embedding_method, 'option': embedding_method_option, "Not attendant": scores}]
					else:
						# Answer for 5, 10, 15, 20 clusters
						result +=  [{'method': embedding_method, 'option': embedding_method_option,'5': scores[3], '10': scores[8], '15': scores[13], '20': scores[18]}]

		return result
