# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:03:46 2018

@author: Administrator
"""

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import multiprocessing
import numpy as np
import jieba
jieba.load_userdict("lstm_data/word.txt")

class W2V_MODEL(object):
    def __init__(self,vocab_dim=100,
                 min_count=1,
                 window_size=5,
                 n_iterations=1,
                 epochs= 50,
                 string_data=None,
                 save_path='lstm_data/Word2vec'):
        self.vocab_dim = vocab_dim
        self.min_count = min_count
        self.window_size = window_size
        self.n_iterations = n_iterations
        self.epochs = epochs
        self.cpu_count = multiprocessing.cpu_count()
        self.string_data = string_data
        self.save_path = save_path
        
        self.word2vec_train()
    def tokenizer(self,text):
        text = [list(document) for document in text] 
#        text = [jieba.lcut(document) for document in text]
        return text
    
    def create_dictionaries(self,model=None):
        if  model is not None:
            gensim_dict = Dictionary()
            gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
            w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
            w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量        
            return w2indx, w2vec
        else:
            print ('No data provided...')
            
    def word2vec_train(self):
        print('Cuting word...')
        combined = self.tokenizer(self.string_data)
        model = Word2Vec(size=self.vocab_dim,
                         min_count=self.min_count,
                         window=self.window_size,
                         workers=self.cpu_count,
                         iter=self.n_iterations)
        model.build_vocab(combined)
        print('Training W2V...')
        model.train(combined,total_examples=len(combined),epochs=self.epochs)
        model.save(self.save_path+'.pkl')
        index_dict, word_vectors = self.create_dictionaries(model=model)
        index_dicted= sorted(index_dict.items(), key=lambda d:d[1], reverse = False) 
        print('Saving as TXT...')
        f = open(self.save_path+'.txt','w',encoding='utf-8')
        for dic in index_dicted:
            string = ''
            for i in range(len(word_vectors[dic[0]])):
                string = string + ' ' +str(word_vectors[dic[0]][i])
            f.write(dic[0] + ' ' + string + '\n')
        a=np.random.normal(size=self.vocab_dim,loc=0,scale=0.05)
        string=''
        for i in range(len(a)):
            string = string + ' ' +str(a[i])
        f.write('UNK' +' ' + string + '\n' )
        f.close
        print('W2V done...')
        return   