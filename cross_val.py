# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:30:45 2017

@author: Jeetu
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def k_fold_crossval(model,x_train,y_train,K=None):
	if K==None:
		K=3
	f1score=[]   
	kf = KFold(n_splits = K, shuffle = True)
	for result in kf.split(y_train):
		X_train,Y_train= x_train.iloc[result[0]],y_train.iloc[result[0]]
		X_test,Y_test = x_train.iloc[result[1]],y_train.iloc[result[1]]
		   
		model.fit(X_train,Y_train)    
		Y_pred=model.predict(X_test)
		cnf_matrix = confusion_matrix(Y_test, Y_pred)
		f1= f1_score(Y_test, Y_pred)
		f1score.append(f1)
	return np.mean(f1score)
	



            
	




