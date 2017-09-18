# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:30:45 2017

@author: Jeetu
"""

from sklearn.model_selection import cross_val_score


class Cross_Valid: 
	def __init__(self,model,x_data,y_data,params=None): 
      self.model=model  
		self.x_data = x_data
		self.y_data = y_data
		self.params=params


	def cross_val(self,x_train,y_train,params=None):
		if params==None:
			params={'cv':5}
		scores = cross_val_score(self.model, .data, iris.target, cv=5)
		return scores