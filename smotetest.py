# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:01:32 2017

@author: Jeetu
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import os 
from data import fetch_data,feature_engineering
from sklearn import preprocessing

root=os.getcwd()
x_train,y_train,x_test,y_test = fetch_data(root,remove_duplicates=True,binary=False)
x_train = feature_engineering(x_train,do_normalization=False)
l=list(x_train)
x_test = feature_engineering(x_test,do_normalization=False)
print('Shape of training data after feature engineering is {}'.format(x_train.shape))
print ('Shape of test data after feature engineering is {}'.format(x_test.shape))
(y_train).value_counts().plot.barh()
maxcount=y_train.value_counts().max()
mincount=y_train.value_counts().min()

le=preprocessing.LabelEncoder()

y_train=le.fit_transform(y_train)+1

datax=pd.DataFrame()
y=pd.Series()


for i in set(y_train):
    y_trainl=(y_train==i)*i
    num=(y_train==i).sum()
    r=maxcount/(145583-num)
    if num>6:
        nbor=5
    else:
        nbor=num-1
    print(pd.Series(y_trainl).value_counts())
    print(i,' current ratio:',num/(145583-num),' expected ratio: ',maxcount/(145583-num),' ',num)    
    if num<maxcount:
        sampler = SMOTE(kind="borderline2",ratio=maxcount/(145583-num),k_neighbors=int(nbor))
        sampled_X,sampled_Y = sampler.fit_sample(x_train,y_trainl)        
    else:
        sampled_X,sampled_Y =x_train,y_trainl
    sampled_X=pd.DataFrame(sampled_X[sampled_Y==i])
    sampled_X.columns = l
    sampled_Y=pd.Series(sampled_Y[sampled_Y==i])
    datax=datax.append(sampled_X)
    y=y.append(sampled_Y)
    print(i,' ',maxcount/(145583-num),' ',num)

y_train_smote=pd.Series(le.inverse_transform(y-1)) 

y_train_smote=y_train_smote.reset_index()
datax=datax.reset_index()


y_train_smote.to_csv('train_smote_label.csv')
datax.to_csv('train_smote_data.csv')


  
    