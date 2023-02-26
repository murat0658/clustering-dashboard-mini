import numpy as np
import os, re
from subprocess import Popen
import json

from callbacks2.pipeline import Pipeline
from callbacks2.TrialsDAO import TrialsDAO

class BertPipeline(Pipeline):
	def __init__(self, sentences_path, search_space = [], int_params = []):
		super(BertPipeline, self).__init__(TrialsDAO("bert", 'sentence-avg'))
		self.search_space = search_space
		self.int_params = int_params
		self.paper_names = []
		self.sentences_path = sentences_path
		self.init_data()
		self.init_sentences()

	def init_data(self):
		for filename in self.filepaths:
			with open(filename, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			self.paper_names.append(paper_name)

	def init_sentences(self):
		if(os.path.exists(self.sentences_path) == False):
			all_sentences = []
			for filename in self.filepaths:
				with open(filename, 'r') as fd:
					fileData = fd.read()
				sentences = re.split('\n|\. ', fileData)
				sentences = list(filter(lambda a: a.strip() != "" and len(a) > 1, sentences))
				sentences.append("\n")
				all_sentences += sentences
			all_sentences_str = "\n".join(all_sentences)
			with open(self.sentences_path, "w") as fd:
				fd.write(all_sentences_str)


	def train(self, params):
		def change_config_file(params):
			with open("bert/bert_dir/uncased_L-2_H-128_A-2/bert_config.json", "r") as fd:
				config_params = json.loads(fd.read())
			config_params['attention_probs_dropout_prob'] = params['attention_probs_dropout_prob']
			with open("bert/bert_config.json", "w") as fd:
				fd.write(json.dumps(config_params))
		def realize_train():
			wd = os.getcwd()
			os.chdir("bert")
			command = "python3 extract_features.py --input_file=tmp/input.txt --output_file=tmp/output.jsonl --vocab_file=bert_dir/uncased_L-2_H-128_A-2/vocab.txt --bert_config_file=bert_dir/uncased_L-2_H-128_A-2/bert_config.json --init_checkpoint=bert_dir/uncased_L-2_H-128_A-2/bert_model.ckpt --layers=-1,-2 --max_seq_length=128 --batch_size=8 > log2.txt"
			command_lst = command.split()
			p = Popen(command_lst) # run the process
			p.wait()
			os.chdir(wd)
		
		def load_embeddings_lst():
			doc_embeddings = []
			doc_embeddings.append(np.zeros(128))
			count = 0
			with open('bert/tmp/output.jsonl', 'r') as f:
				for line in f:
					sentence_vals = json.loads(line)
					features = sentence_vals["features"]
					if (len(features) == 2 and features[0]["token"] == '[CLS]' and features[1]["token"] == '[SEP]'):
						try:
							line2 = next(f)
							sentence_vals = json.loads(line2)
							features = sentence_vals["features"]
							if (len(features) == 2 and features[0]["token"] == '[CLS]' and features[1]["token"] == '[SEP]'):
								count += 1
								print(str(count*100/600.0) + "%")
								doc_embeddings.append(np.zeros(128))
						except StopIteration:
							print("Bert model loaded")
							return doc_embeddings
					sentence_feature_embedding = np.zeros(128)
					for feature in features:
						feature_embedding = np.array(feature["layers"][0]["values"]) + np.array(feature["layers"][1]["values"])
						sentence_feature_embedding += feature_embedding
						sentence_feature_embedding = np.divide(sentence_feature_embedding, len(features) * 2)
					doc_embeddings[-1] += sentence_feature_embedding
			return doc_embeddings
		
		def give_titles_to_vectors(doc_embeddings_list):
			doc_embeddings_dict = {}
			for (i, title) in enumerate(self.paper_names):
				doc_embeddings_dict[title] = list(doc_embeddings_list[i])
			return doc_embeddings_dict

		change_config_file(params)
		realize_train()
		doc_embeddings_list = load_embeddings_lst()
		return give_titles_to_vectors(doc_embeddings_list)