from hyperopt import Trials
from pymongo import MongoClient, DESCENDING, ASCENDING
import pickle
from bson import BSON
import gridfs

from constants import connection_string, dbname, scores_to_params

class TrialsDAO:
	def __init__(self, collection_name, method_option=None):
		client = MongoClient(connection_string)
		self.fs = gridfs.GridFS(client[dbname]) 
		self.collection = client[dbname][collection_name]
		if method_option != None:
			self.method_option = method_option

	def getTrialsFromDb(self):
		print("method_option: ", self.method_option)
		trialsCursor = self.collection.find(
		        {   'trials': { '$exists': True},
		            'method_option': self.method_option
		        }
		    )
		if trialsCursor.count() == 0:
			return (None, Trials())
			print("trials object is initialized")
		elif trialsCursor.count() == 1:
			return (trialsCursor[0]['_id'], pickle.loads(trialsCursor[0]['trials']))
			print("trials object is initialized")
		else:
			raise Exception('There is something wrong with trials file in db.')

	def saveTrialsToDb(self, trials, _id = None):
		print("saving trials to db")
		if(_id == None):
			self.collection.insert_one({ 'trials': pickle.dumps(trials), 'method_option': self.method_option })
		else:    
			self.collection.update_one({'_id': _id}, {'$set': { 'trials': pickle.dumps(trials), 'method_option': self.method_option }})

	def save_document(self, paper_embedding_dict):
		print("saving document")
		paper_embedding_dict['method_option'] = self.method_option
		self.collection.insert_one(paper_embedding_dict)

	def save_big_document(self, paper_embedding_dict, embeddings):
		print("saving big document")
		paper_embedding_dict['method_option'] = self.method_option
		# first save the embeddings
		grid_id = self.fs.put(BSON.encode(embeddings))
		paper_embedding_dict['embeddings'] = grid_id
		# then save the rest
		self.collection.insert_one(paper_embedding_dict)

	def get_optimized_params(self, index = 'rand_index', clustering_method ='hierarthical_scores'):
		only_cluster_info = { '$match': {'trials': { '$exists': False},  'method_option': self.method_option } }
		params_column = '$optimized_scores.' + scores_to_params[clustering_method] + '.' + index + '_param'
		scores_column = '$optimized_scores.' + clustering_method + '.' + index
		
		projection = {'$project': 
		    {
		        'average_score':  {
		            '$avg': scores_column
		        },
		        'optimized_params':  params_column,
		    }
		}

		sort_avg_score_descending = {
		   "$sort": {  'average_score': DESCENDING }
		}
		if index == 'davies_bouldin_score':
			sort_avg_score_descending = {
				"$sort": {  'average_score': ASCENDING }
		}

		limit_1 = { "$limit": 1 }
		pipeline = [only_cluster_info, projection, sort_avg_score_descending, limit_1]
		return list(self.collection.aggregate(pipeline))

	def get_document(self, index = 'rand_index', clustering_method ='hierarthical_scores'):
		print("index: ", index, "clustering_method: ", clustering_method)
		only_cluster_info = { '$match': {'trials': { '$exists': False},  'method_option': self.method_option } }
		scores_column = '$optimized_scores.' + clustering_method + '.' + index
		projection = {'$project': 
		    {
		        'average_score':  {
		            '$avg': scores_column
		        },
		        'optimized_scores':  scores_column,
		        'embeddings':  '$embeddings'
		    }
		}

		sort_avg_score_descending = {
		   "$sort": {  'average_score': DESCENDING }
		}
		if index == 'davies_bouldin_score':
			sort_avg_score_descending = {
				"$sort": {  'average_score': ASCENDING }
		}

		limit_1 = { "$limit": 1 }
		pipeline = [only_cluster_info, projection, sort_avg_score_descending, limit_1]
		res= list(self.collection.aggregate(pipeline))
		if(res != None and len(res) > 0 and type(res[0]['embeddings']) != dict):
			print(len(res))
			print("res 0: ", res[0].keys())
			res[0]['embeddings'] = BSON.decode(self.fs.get(res[0]['embeddings']).read())
			print('length of embeddings: ',len(res[0]['embeddings']))
		return res

	def get_embeddings(self, index = 'silhouette_score', clustering_method ='hierarthical_scores'):
		document = self.get_document(index, clustering_method)
		if(document != []):
			return document[0]['embeddings']
		else:
			return {}


	def get_all_docs(self):
		return list(self.collection.find( {'trials': { '$exists': False}}))

	def update_one_doc(self, condition, data):
		self.collection.update_one(condition, data)


