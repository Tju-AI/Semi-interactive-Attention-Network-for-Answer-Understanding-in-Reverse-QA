# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:00:17 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
#import re
from keras.models import load_model
from Layer_one import MyLayer_one
from Layer_crf import CRF
import jieba 
import os
jieba.load_userdict("lstm_data/word.txt")

vocab_dim = 100
maxlen_context = 30


def load_train_file(path):
    merge_data=[]
    for file in os.walk(path):
        for filename in file[2]:
            child = os.path.join(path,str(filename))
            da=pd.read_excel(child,index=None)
            merge_data.append(da)
    data = pd.concat(merge_data,ignore_index=True)  
    context = list(data['question'])
    target = list(data['target'])
    answer = list(data['answer'])

    label = np.array(list(data['label']))
    return context,target,answer,label

def get_index(path):
    f = open(path, 'r',encoding='utf-8')
    index_dict = {}
    word_vectors ={}
    word2vec = f.readlines()
    for i in range(len(word2vec)):
        one = word2vec[i].split()
        index_dict[one[0]]=i+1
        vec = np.array(one[1:],dtype='float32')
        word_vectors[one[0]] = vec
    f.close()
    return index_dict,word_vectors
def feature_embeding(comment):
    size=15
    par=1
    data=pd.read_excel('lstm_data/feature_word.xlsx',index=None)
    definite_words = list(data['肯定词'])
    positive_words = list(data['正向'])
    negative_words = list(data['负向'])
    imagine_words = list(data['假想词'])
    deny_words = list(data['否定词'])
    inter_words = list(data['疑问词'])
    assume_words = list(data['假定词'])
    feature_embed = np.zeros((len(comment),maxlen_context,1*size))
    for i,t in enumerate(comment):
        token = jieba.tokenize(t[:maxlen_context])
        for tk in token:
            if tk[0] in deny_words:
                feature_embed[i,tk[1]:tk[2],0:size]= par
            if tk[0] in inter_words:
                feature_embed[i,tk[1]:tk[2],size:2*size]= par
            if tk[0] in assume_words:
                feature_embed[i,tk[1]:tk[2],2*size:3*size]= par
            if tk[0] in definite_words:
                feature_embed[i,tk[1]:tk[2],3*size:4*size]= par
            if tk[0] in positive_words:
                feature_embed[i,tk[1]:tk[2],4*size:5*size]= par
            if tk[0] in negative_words:
                feature_embed[i,tk[1]:tk[2],5*size:6*size]= par
            if tk[0] in imagine_words:
                feature_embed[i,tk[1]:tk[2],6*size:7*size]= par                
    return feature_embed
def target_embeding(comment,target):
    size=15
    par=0.3
    target_embed = np.zeros((len(comment),maxlen_context,size))
    for i,t in enumerate(comment):
        ind=t.index(target[i])
        target_embed[i,ind:(ind+len(target[i])),0:size]= par          
    return target_embed
def embedding(string_data):
    def tokenizer(text):
        text = [list(document) for document in text] 
#        text = [jieba.lcut(document) for document in text]
        return text
    def parse_dataset(combine):
        ''' Words become integers
        '''
        index_dict,word_vectors = get_index('lstm_data/word2vec_ci.txt')
        data=[]
        for sentence in combine:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(index_dict[word])
                except:
                    new_txt.append(index_dict['UNK'])
            data.append(new_txt)
        return data
    def zero_pad(X,seq_len):
        return np.array([x[:seq_len - 1]+[0] * max(seq_len - len(x), 1)  for x in X])
    

    list_word = tokenizer(string_data)
    combine=parse_dataset(list_word)
    combined= zero_pad(combine,maxlen_context)
    return combined

def predict_one(context,target):
    context_combined = embedding(context)
    feature_embed_q = feature_embeding(context)
    target_combined = embedding(target)
    feature_embed_ans = feature_embeding(target)
    score = model.predict([context_combined,feature_embed_q,target_combined,feature_embed_ans])
    classes = np.argmax(score,axis=1)
#    print(score)
    return classes
def test(model,context,target,answer,label):
    context_combined = embedding(context)
    feature_embed_q = target_embeding(context,target)
    target_combined = embedding(answer)
    feature_embed_ans = feature_embeding(answer)
    score = model.predict([context_combined,feature_embed_q,target_combined,feature_embed_ans])
    classes = np.argmax(score,axis=1)
    result_bool = np.equal(label, classes)
    true_num = np.sum(result_bool)
    acc = true_num/len(result_bool)
    prob=[score[i][classes[i]] for i in range(len(classes))]
#    print("The accuracy of the model is %f" % (true_num/len(result_bool)))
    return classes,acc,prob

def evaluate():
    
    model= load_model('lstm_model/part_1_relation.h5',custom_objects={"MyLayer_one": MyLayer_one})
    question,target,answer,label = load_train_file('data')
    test_question,test_target,test_answer,test_label = load_train_file('test')
    train_result,train_acc,train_prob=test(model,question,target,answer,label)
    test_result,test_acc,test_prob=test(model,test_question,test_target,test_answer,test_label)
    test_result2,test_acc2,test_prob2=test(model,test_question,test_target,test_answer,test_label)
    print("训练集准确率 %f" % (train_acc))
    print("测试集准确率 %f" % (test_acc))
    return train_result,test_result,train_acc,test_acc,train_prob,test_prob




