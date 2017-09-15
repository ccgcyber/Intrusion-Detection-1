"""
Thought is to import this file into a jupyter notebook 
and call models from here 
so in the jupyter notebook we load the dataset and
do modelling from here.

This class is expecting a dataframe 

"""
import xgboost as xgb 
import pandas as pd 
from sklearn.model_selection import GridSearchCV as gcv 
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

class Model(): 
	def __init__(self,data,labelname): 
		self.data = data 
		ncol=
		y_train=data.loc[:,labelname]
		x_train = data[-y_train]
		self.x_train = x_train
		self.y_train = y_train
		

	def xgboost(self,x_train,y_train,x_test):
		x_dtrain=xgb.DMatrix(data=self.x_train,labels=self.y_train)
		x_dtest=xgb.DMatrix(data=self.x_test)

		params ={'eta': 1,'max_depth':2 ,
		'objective':'binary:logistic'
		}
		num_round = 4
		model = xgb.train(params,x_dtrain,num_round)
		return model


	def mnbayes(x_train,y_train):
		clf = MultinomialNB()
		model=clf.fit(x_train,y_train)
		return model

	def svm(x_train ,y_train):
		svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
		#for svm gamma and c need to be learnt using grid search
		model=svm.fit(X_train, y_train)
		return model

	def randforest(x_train,y_train):
		forest= RandomForestClassifier(criterion='entropy',max_depth=20,
									   n_estimators = 100,random_state = 1,n_jobs = 2)
		#parameters n_estimators ,max_depth need to be learnt
		model=forest.fit(X_train, y_train)
		return model





