"""
Thought is to import this file into a jupyter notebook 
and call models from here 
so in the jupyter notebook we load the dataset and
do modelling from here.

 

"""
import xgboost as xgb 
import pandas as pd 
from sklearn.model_selection import GridSearchCV as gcv 
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class Model: 
	def __init__(self,x_train,y_train,params=None): 
		self.x_train = x_train
		self.y_train = y_train
		self.params=params
		

	def xgboost(self,x_train,y_train,x_test,params=None):
		x_dtrain=xgb.DMatrix(data=self.x_train,labels=self.y_train)
		x_dtest=xgb.DMatrix(data=self.x_test)
		if params == None :
			params ={'eta': 1,'max_depth':2 ,
			'objective':'binary:logistic'
			}
		num_round = 4
		model = xgb.train(params,x_dtrain,num_round)
		return model


	def mnbayes(self,x_train,y_train):
		model = MultinomialNB()
		model.fit(x_train,x_test)		
		return model

	def svm(self,x_train ,y_train,params=None):
		if params == None:
			params={'kernel':'rbf','random_state':0,'gamma':0.1,
			'C':10}
		svm = SVC()
		svm.set_params(**params)
		#for svm gamma and c need to be learnt using grid search
		svm.fit(x_train,y_train)		
		return svm

	def randforest(self,x_train,y_train,params=None):
		if params == None:
			params={'criterion':'entropy','max_depth':20,'n_estimators':100,
		'random_state':12345,'n_jobs':2}
		forest= RandomForestClassifier()
		forest.set_params(**params)
		#parameters n_estimators ,max_depth need to be learnt
		forest.fit(x_train,y_train)		
		return forest





