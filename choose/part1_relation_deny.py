# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:01:54 2018

@author: Administrator
"""

from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import re
from keras.models import load_model
from w2v_model import W2V_MODEL
from keras.layers import Bidirectional,Input,GlobalAveragePooling1D,Flatten,Add
from keras.layers import TimeDistributed,Permute,merge,Lambda,concatenate,Reshape,RepeatVector
from keras.layers.core import Dense, Dropout,Activation
from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM 
from keras import callbacks
from Layer_one import MyLayer_one
from keras import losses
from test_part_deny import evaluate
import jieba
import os
import xlwt
jieba.load_userdict("lstm_data/word.txt")

vocab_dim = 100
maxlen_context = 30
batch_size = 32
n_epoch = 50


def load_w2v_file(path):
    data=pd.read_excel(path,index=None)
    word_souce = list(data['comment'])
    word_train = [re.sub(' |　| ', '', sample, count=0, flags=0) for sample in word_souce ]
    return word_train

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

def embedding(string_data):
    def tokenizer(text):
        text = [list(str(document)) for document in text] 
        #    text = [jieba.lcut(document) for document in text]
        return text
    def parse_dataset(combine):
        ''' Words become integers
        '''
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

def split_data(x1,x2,y,test_size):
    x1_train,x1_test,x2_train,x2_test,y_train,y_test = train_test_split(x1, x2, y,test_size=test_size)
    return x1_train,x1_test,x2_train,x2_test,y_train,y_test

def get_weight():
    embedding_weights = np.zeros((len(index_dict) + 1,vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
#        print(index)
        embedding_weights[index,:] = word_vectors[word]
    return embedding_weights
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
#        print(i,t)
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
def create_base_network(input_shape,input_shape2):
    '''Base network to be shared (eq. to feature extraction).
    '''
    a = Input(shape=input_shape)
    fea_emb = Input(shape=input_shape2)
    word_emb = Embedding(output_dim=vocab_dim,
              input_dim=len(index_dict) + 1,
              mask_zero=False,
              weights=[get_weight()],
              input_length=maxlen_context,
              trainable=False)(a)
    embed_con = concatenate([word_emb,fea_emb])
    print('embed:',np.shape(embed_con))
    x = Bidirectional(LSTM(units=50,return_sequences=True))(embed_con)
    print(np.shape(x))
    hidden = TimeDistributed(Dense(100, activation='tanh'))(x)
    print(np.shape(hidden))
    return Model([a,fea_emb], hidden)
	
def train_lstm(x1_train,x1_test,f1_train,f1_test,x2_train,x2_test,f2_train,f2_test,y_train,y_test):
    print('Defining a Simple Keras Model...')
    input_shape = x1_train.shape[1:]
    input_shape2 = f1_train.shape[1:]
    base_network = create_base_network(input_shape,input_shape2)
    Mydot = Lambda(lambda x: K.batch_dot(x[0],x[1]))
    crf = CRF(2, sparse_target=True,learn_mode ='marginal')
    Get_one = Lambda(lambda x: x[:,:,0])
	
    #Q-part bi-lstm Input
    input_con = Input(shape=input_shape)
    input_f1 = Input(shape=input_shape2)
    hid_con = base_network([input_con,input_f1])

    ave_con = GlobalAveragePooling1D()(hid_con)
    print('ave_con:',np.shape(ave_con))
    ave_con = Reshape((100,1))(ave_con)
    print('avg_con_reshape:',np.shape(ave_con))
	
    #A-part bi-lstm Input
    input_tag = Input(shape=input_shape)
    input_f2 = Input(shape=input_shape2)
    hid_tag = base_network([input_tag,input_f2])

    ave_tag = GlobalAveragePooling1D()(hid_tag)
    print('avg_tag:',np.shape(ave_tag))
    ave_tag = Reshape((100,1))(ave_tag)
    print('avg_tag_reshape:',np.shape(ave_tag))
   
    #A-part Attention
    tag_at = MyLayer_one()([hid_tag,ave_con])
    tag_at = Flatten()(tag_at)
    tag_at = Activation('softmax')(tag_at)
    tag_at = RepeatVector(1)(tag_at)

    print('tag_at:',np.shape(tag_at))
    att_tag_mul = Mydot([tag_at, hid_tag])
    print('att_tag_mul:',np.shape(att_tag_mul))
     
    at_done = Flatten()(att_tag_mul)
    print('out:',np.shape(at_done))
    output = Dropout(0.1)(at_done)
    output = Dense(3)(output)
    output = Activation('softmax')(output)
    model = Model(inputs=[input_con,input_f1,input_tag,input_f2], outputs=output)
    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    print("Train...")
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    saveBestModel = callbacks.ModelCheckpoint('lstm_model/part_1_relation.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    model.fit([x1_train,f1_train,x2_train,f2_train], y_train, batch_size=batch_size, epochs=n_epoch,verbose=1, 
              validation_data=([x1_test,f1_test,x2_test,f2_test], y_test),callbacks=[earlyStopping, saveBestModel])    
    

if __name__=='__main__':
    r = []
    for i in range(11):
        if i<10:
            print('Loading Train Data...')
            context,target,answer,label = load_train_file('data')
            print(len(context))
            index_dict,word_vectors = get_index('lstm_data/word2vec_ci.txt')
            label= to_categorical(label, num_classes=None) 
            context_combined = embedding(context)
            target_embed_q = target_embeding(context,target)
            answer_combined = embedding(answer)
            feature_embed_ans = feature_embeding(answer)
            
            x1_train,x1_test,f1_train,f1_test,x2_train,x2_test,f2_train,f2_test,y_train,y_test = train_test_split(context_combined,target_embed_q,answer_combined,feature_embed_ans,label,test_size=0.2)
            train_lstm(x1_train,x1_test,f1_train,f1_test,x2_train,x2_test,f2_train,f2_test,y_train,y_test)
            print('evaluating...')
            train_result,test_result,train_acc,test_acc,train_prob,test_prob = evaluate()
            r.append(test_acc)
    print(r)
#    a = 0
#    for i in range(len(r)):
#       a = a+r[i]
#    b = a/20
#   print(b)
            

    
    
    
    
    
    
    
    