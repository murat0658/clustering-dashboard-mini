from callbacks2.AfterBertPipeline import AfterBertPipeline
from callbacks2.AfterBertPipeline2 import AfterBertPipeline2
from callbacks2.Doc2vecPipeline import Doc2vecPipeline
from callbacks2.FasttextPipeline import FasttextPipeline
from callbacks2.FasttextTfIdfPipeline import FasttextTfIdfPipeline
from callbacks2.GlovePipeline import GlovePipeline
from callbacks2.GloveTfIdfPipeline import GloveTfIdfPipeline
from callbacks2.GlovePretrainedPipeline import GlovePretrainedPipeline
from callbacks2.BertPipeline import BertPipeline
from callbacks2.SentenceBertPipeline import SentenceBertPipeline
from callbacks2.SkipThoughtsPipeline import SkipThoughtsPipeline
from callbacks2.Word2VecPipeline import Word2VecPipeline
from callbacks2.Word2VecTfIdfPipeline import Word2VecTfIdfPipeline

from search_spaces import *

def runMethod(embedding_method, embedding_method_option, manuel_params = None):
    if(embedding_method == "roberta"):
        if(embedding_method_option == 'sentence-avg'):
            AfterBertPipeline(["roberta", "roberta-base", "roberta"], roberta_gpt2_search_space, roberta_gpt2_int_params).run(manuel_params)
        if(embedding_method_option == 'sentence-avg2'):
            AfterBertPipeline2(["roberta", "roberta-base", "roberta"], roberta_gpt2_search_space, roberta_gpt2_int_params).run(manuel_params)            
    elif(embedding_method == "gpt2"):
        if(embedding_method_option == 'sentence-avg'):
            AfterBertPipeline(["gpt2", "gpt2", "gpt2"], roberta_gpt2_search_space, roberta_gpt2_int_params).run(manuel_params)
        elif(embedding_method_option=='sentence-avg2'):
            AfterBertPipeline2(["gpt2", "gpt2", "gpt2"], roberta_gpt2_search_space, roberta_gpt2_int_params).run(manuel_params)
    elif(embedding_method == "scibert"):
        if(embedding_method_option == 'sentence-avg'):
            AfterBertPipeline(["bert", "allenai/scibert_scivocab_uncased", "scibert"], roberta_gpt2_search_space, roberta_gpt2_int_params).run(manuel_params)
        elif(embedding_method_option == 'sentence-avg2'):
            AfterBertPipeline2(["bert", "allenai/scibert_scivocab_uncased", "scibert"], roberta_gpt2_search_space, roberta_gpt2_int_params).run(manuel_params)
    elif(embedding_method == "skip thoughts"):
        SkipThoughtsPipeline().run()
    elif(embedding_method == "doc2vec"):
        Doc2vecPipeline(doc2vec_search_space, doc2vec_int_params).run( manuel_params)
    elif embedding_method == 'fasttext' : #burası yeni
        if embedding_method_option == 'avg':
            FasttextPipeline(fasttext_search_space, fasttext_int_params).run(manuel_params)
        elif embedding_method_option == 'tf-idf':
            FasttextTfIdfPipeline(fasttext_search_space, fasttext_int_params).run(manuel_params)
    elif embedding_method == 'glove':
        if(embedding_method_option == 'avg'):
            GlovePipeline(glove_search_space, glove_int_params).run(manuel_params)
        elif embedding_method_option == 'tf-idf':
            GloveTfIdfPipeline(glove_search_space, glove_int_params).run(manuel_params)
        elif embedding_method_option == 'pre-prepared-embeddings':
            GlovePretrainedPipeline(glove_pretrained_search_space).run(manuel_params)
    elif embedding_method == 'bert':
        BertPipeline("cappadocia_papers_sentences.txt", bert_search_space).run(manuel_params)
    elif embedding_method == 'sentence bert':
        SentenceBertPipeline(sentence_bert_search_space, sentence_bert_int_params).run(manuel_params)
    elif embedding_method == 'word2vec':
        if embedding_method_option == 'avg':
            Word2VecPipeline(word2vec_search_space, word2vec_int_params).run(manuel_params)
        elif embedding_method_option == 'tf-idf':
            Word2VecTfIdfPipeline(word2vec_search_space, word2vec_int_params).run(manuel_params)
    else:
        raise Exception("No such method exists")
    best_data = []#get_scores()
    return [True, True, best_data]
