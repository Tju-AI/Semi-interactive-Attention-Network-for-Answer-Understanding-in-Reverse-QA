# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 09:06:37 2018

@author: Administrator
"""


import pandas as pd
from sklearn.utils import shuffle
from w2v_model import W2V_MODEL
import re
import os 
filtrate = re.compile(u'[^\u4E00-\u9FA50-9a-zA-Z，。,.？!?！:：;；]')
def load_w2v_file(path):
    merge_data=[]
    for file in os.walk(path):
        for filename in file[2]:
            child = os.path.join(path,str(filename))
            da=pd.read_excel(child,index=None)
            merge_data.append(da)
    data = pd.concat(merge_data,ignore_index=True)
    comment = list(data['answer'])
    comment_cl = []
    for t in comment:
        b = re.sub(' ', ',', t, count=0, flags=0)
        b = filtrate.sub(r'', b)
        comment_cl.append(b)
    comment_cl = shuffle(comment_cl)
    return comment_cl

def train_w2v(coment):
    W2V_MODEL(vocab_dim=100,
              min_count=5,
              window_size=5,
              n_iterations=1,
              epochs= 50,
              string_data=coment,
              save_path='lstm_data/word2vec_ci')

comment= load_w2v_file('data')
train_w2v(comment)