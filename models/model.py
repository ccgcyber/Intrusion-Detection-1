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
from sklearn.naive_bayes import GaussianNB


class Model(): 
	def __init__(self,data,labelname): 
		self.data = data 
		ncol=
		y_train=data.loc[:,labelname]
		x_train = data[-y_train]
		self.x_train = x_train
		self.y_train = y_train
		

	def xgboost(x_train,y_train,x_test):
		x_dtrain=xgb.DMatrix(data=x_train,labels=y_train)
		x_dtest=xgb.DMatrix(data=x_test)

		params ={'eta': 1,'max_depth':2 ,
		'objective':'binary:logistic'
		}
		num_round = 4
		model = xgb.train(params,x_dtrain,num_round)
		return model
    
   def gnbayes(x_train,y_train,x_test):
      model=gnb.fit(x_train, y_train)
      return model
        





