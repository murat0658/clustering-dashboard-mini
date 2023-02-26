import stanza, os
import pickle, string
from nltk.corpus import stopwords
import pandas as pd
from callbacks2.pipeline import Pipeline

from constants import initial_files_path

class Tokenizer():
	def __init__(self):
		self.nlp = stanza.Pipeline(lang='en', processors='tokenize')
		self.token_info = {}
		self.filepaths = Pipeline().filepaths

	def tokenize(self):
		token_counts = {}
		token_papers = {}
		for filepath in self.filepaths:
			with open(filepath, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			doc = self.nlp(data)
			for sentence in doc.sentences:
				for token in sentence.tokens:
					txt = token.text
					if(txt in token_counts):
						token_counts[txt] += 1
						token_papers[txt].add(paper_name)
					else:
						token_counts[txt] = 1
						token_papers[txt] = {paper_name}
		token_counts = self.clean_data(token_counts)
		with open(initial_files_path + '/token_info', "wb") as fd:
			pickle.dump({'token_papers': token_papers, 'token_counts': token_counts}, fd)

	def clean_data(self, token_counts):
		# sort it in descending order
		token_counts = dict(sorted(token_counts.items(), key=lambda item: item[1], reverse=True))
		# clean the punctuation
		token_counts = dict(filter(lambda item: item[0] not in string.punctuation,token_counts.items()))
		# clean the numbers
		token_counts = dict(filter(lambda item: item[0].isdigit() == False, token_counts.items()))
		# clean the stop words
		eng_stopwords = stopwords.words('english')
		token_counts = dict(filter(lambda item: item[0].lower() not in eng_stopwords, token_counts.items()))
		#clean unnecessary words that i see 
		unnecessary_words = ['“', '’', '’s', 'et', 'al.', 'Figure', 'Fig.', '‘', '–',  '”', '•', "'s", 'p.', 'i.e.', 'etc.', 'also']
		token_counts = dict(filter(lambda item: item[0] not in unnecessary_words, token_counts.items()))
		return token_counts

	def create_bi_gram_words(self, token_papers):
		bigram_list = []
		for filepath in self.filepaths:
			with open(filepath, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			data_list = data.split()
			if(len(data_list) > 0):
				for (i,word) in enumerate(data_list):
					if(i+1 < len(data_list)):
						bigram_word = word + " " + data_list[i+1]
						if(bigram_word in token_papers):
							if paper_name not in token_papers[bigram_word]:
								token_papers[bigram_word].append(paper_name)
						else:
							token_papers[bigram_word]  = [paper_name]
							bigram_list.append(bigram_word)
		return bigram_list

	def get_token_papers(self):
		with open(initial_files_path + '/token_info', "rb") as fd:
			token_info = pickle.load(fd)
		return token_info['token_papers']

	def get_token_papers_with_bigram_words(self):
		res = self.get_token_papers()
		self.create_bi_gram_words(res)
		return res

	def get_word_list(self):
		token_info_path = initial_files_path + '/token_info';
		if os.path.isfile(token_info_path):
			with open(token_info_path, "rb") as fd:
				token_info = pickle.load(fd)
			token_counts = token_info['token_counts']
			word_list = [tup[0] for tup in list(token_counts.items())]
			word_list += self.create_bi_gram_words(token_info['token_papers'])
			return word_list
		return []

	def create_word_dict(self):
		word_list = self.get_word_list()
		return [{"label": word, "value": word} for word in word_list]

	def create_word_data_frame(self):
		token_info_path = initial_files_path + '/token_info';
		if os.path.isfile(token_info_path):
			with open(token_info_path, "rb") as fd:
				token_info = pickle.load(fd)
			token_counts = token_info['token_counts']
			df = pd.DataFrame(token_counts.items()) 
			df.columns = ["word", "count"]
			return df
		return None
