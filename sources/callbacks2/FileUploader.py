import io, base64
from zipfile import ZipFile
from callbacks2.Tokenizer import Tokenizer
from callbacks2.pipeline import Pipeline
import os, re
import pickle
import math
import json
import shutil

from constants import initial_files_path

class FileUploader():
	def __init__(self):
		self.documents_path = initial_files_path + '/documents'
		self.delete_unnecessary_mac_folder()
		self.init_filepaths()

	def init_filepaths(self):
		path = initial_files_path + '/documents'
		if os.path.isdir(path) == False: 
			os.makedirs(path)
		pwd = os.getcwd()
		os.chdir(path)
		self.filepaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.txt'];
		os.chdir(pwd)

	def delete_unnecessary_mac_folder(self):
		mac_folder = initial_files_path + '/documents/__MACOSX'
		if os.path.isdir(mac_folder):
			shutil.rmtree(mac_folder, ignore_errors=True)

	def upload_file(self, content):
		self.remove_old_files_and_folders()
		_, content_string = content.split(',')
		# Decode the base64 string
		content_decoded = base64.b64decode(content_string)
		# Use BytesIO to handle the decoded content
		zip_str = io.BytesIO(content_decoded)
		# Now you can use ZipFile to take the BytesIO output
		with ZipFile(zip_str, 'r') as zip_obj:
			try:
				zip_obj.extractall(self.documents_path)
			except Exception as err:
				print("Dosya yükleme hatası: " + str(err))
		self.delete_unnecessary_mac_folder()
		self.init_filepaths()

	def upload_annotation_file(self, content):
		def check_if_annotated_files_exist(data):
			annotated_clusters = json.loads(data)
			annotated_docs = []
			for lst in annotated_clusters.values():
				annotated_docs += lst
			all_docs = []
			for filename in self.filepaths:
				with open(filename, 'r') as fd:
					data = fd.read()
				paper_name = data.split('\n')[0]
				all_docs.append(paper_name)

			for annotated_doc in annotated_docs:
				if(annotated_doc not in all_docs):
					raise Exception('Document does not exist!!!\n' + annotated_doc)

		_, content_string = content.split(',')
		# Decode the base64 string
		content_decoded = base64.b64decode(content_string)
		check_if_annotated_files_exist(content_decoded.decode('utf-8'))
		# Use BytesIO to handle the decoded content
		annotations_filepath = initial_files_path + '/annotations.json'
		print(annotations_filepath)
		with open(annotations_filepath, 'wb') as fd:
			fd.write(content_decoded)

	def write_tf_idf_values_fasttext(self):
		term_freq = {}
		inv_doc_freq = {} #inverse document frequency
		for filepath in self.filepaths:
			with open(filepath, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			term_freq[paper_name] = {}
			data_list = data.split()
			if(len(data_list) > 0):
				for word in data_list:
					if(word in term_freq[paper_name]):
						term_freq[paper_name][word] += 1
					else:
						term_freq[paper_name][word] = 1            
				for word in term_freq[paper_name]:
					if(word in inv_doc_freq):
						inv_doc_freq[word] += 1
					else:
						inv_doc_freq[word] = 1

		for word in inv_doc_freq:
			inv_doc_freq[word] = math.log2(1+(len(self.filepaths) / (1+inv_doc_freq[word])))

		tf_idf_dict = {}
		tf_idf_fasttext_paper_total_dict = {}
		for paper_name in term_freq:
			tf_idf_dict[paper_name] = {}
			tf_idf_fasttext_paper_total_dict[paper_name] = 0
			for word in term_freq[paper_name]:
				tf_idf_value = math.log2(1 + inv_doc_freq[word] * term_freq[paper_name][word])
				tf_idf_dict[paper_name][word] = tf_idf_value
				tf_idf_fasttext_paper_total_dict[paper_name] += tf_idf_value
    
		tf_idf_file = initial_files_path + '/tf_idf_fasttext'
		with open(tf_idf_file, "wb") as fd:
			pickle.dump({"tf_idf_fasttext_dict" : tf_idf_dict, "tf_idf_fasttext_paper_total_dict": tf_idf_fasttext_paper_total_dict}, fd)

	def write_tf_idf_values_glove(self):
		term_freq = {}
		inv_doc_freq = {} #inverse document frequency
		for filepath in self.filepaths:
			
			with open(filepath, 'r') as fd:
				data = fd.read()
			paper_name = data.split('\n')[0]
			term_freq[paper_name] = {}
			data_list = re.split('\s+|\.', data)
			data_list = list(filter(lambda a: a != '', data_list))
			if(len(data_list) == 0):
				break;
			for word in data_list:
				word = re.sub(r'\W+', '', word).lower();
				if(word != ''):
					if(word in term_freq[paper_name]):
						term_freq[paper_name][word] = term_freq[paper_name][word] + 1
					else:
						term_freq[paper_name][word] = 1            
			for word in term_freq[paper_name]:
				if(word in inv_doc_freq):
					inv_doc_freq[word] += 1
				else:
					inv_doc_freq[word] = 1

		for word in inv_doc_freq:
			inv_doc_freq[word] = math.log2(1+(len(self.filepaths) / (1+inv_doc_freq[word])))

		tf_idf_dict = {}
		tf_idf_glove_paper_total_dict = {}
		for paper_name in term_freq:
			tf_idf_dict[paper_name] = {}
			tf_idf_glove_paper_total_dict[paper_name] = 0
			for word in term_freq[paper_name]:
				tf_idf_value = math.log2(1 + inv_doc_freq[word] * term_freq[paper_name][word])
				tf_idf_dict[paper_name][word] = tf_idf_value
				tf_idf_glove_paper_total_dict[paper_name] += tf_idf_value
		tf_idf_file = initial_files_path + '/tf_idf_glove'
		with open(tf_idf_file, "wb") as fd:
			pickle.dump({"tf_idf_glove_dict" : tf_idf_dict, "tf_idf_glove_paper_total_dict": tf_idf_glove_paper_total_dict}, fd)

	def writeFullData(self):
		fullDataFile = initial_files_path + '/fullData.txt'
		with open(fullDataFile, 'w') as fdw:
			for filename in self.filepaths:
				with open(filename, "r") as fd:
					data = fd.read()
				word_list = re.split('\s+|\.', data)
				word_list  = list(filter(lambda a: a != '', word_list))
				for word in word_list:
					word = re.sub(r'\W+', '', word).lower()
					fdw.write(word + " ")
		os.chmod(fullDataFile , 0o777)

	def init_sentences(self):
		sentences_path = initial_files_path + '/sentences.txt';
		all_sentences = []
		for filename in self.filepaths:
			with open(filename, 'r') as fd:
				fileData = fd.read()
			sentences = re.split('\n|\. ', fileData)
			sentences = list(filter(lambda a: a.strip() != "" and len(a) > 1, sentences))
			sentences.append("\n")
			all_sentences += sentences
		all_sentences_str = "\n".join(all_sentences)
		with open(sentences_path, "w") as fd:
			fd.write(all_sentences_str)

	def remove_old_files_and_folders(self):
		if(os.path.isdir(self.documents_path)):
			shutil.rmtree(self.documents_path, ignore_errors=True)
		annotations_filepath = initial_files_path + '/annotations.json'
		if(os.path.isfile(annotations_filepath)):
			os.remove(annotations_filepath)


	def initialize_initial_files_to_train(self):
		self.init_sentences()
		print("sentences initialized...")
		self.write_tf_idf_values_fasttext()
		print("fasttext tf-idf values were calculated...")
		self.write_tf_idf_values_glove()
		print("glove tf-idf values were calculated...")
		Tokenizer().tokenize()
		print("Word grams are created...")
		self.writeFullData()
		print("Initial files were all created...")

