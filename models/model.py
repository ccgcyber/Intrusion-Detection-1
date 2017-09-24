import xgboost as xgb 
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

class Model: 
	def __init__(self,x_train,y_train,params=None): 
		self.x_train = x_train
		self.y_train = y_train
		self.params=params
		

	def xgboost(self,x_train,y_train,x_test,params=None,cv=True):		
		x_dtrain=xgb.DMatrix(data=self.x_train,label=self.y_train)
		x_dtest=xgb.DMatrix(x_test)
		if params == None :
			params ={'eta': 1,'max_depth':5 ,'multi':'softmax',
			'objective':'binary:logistic'
			}
		num_round = 10
		model = xgb.train(params,x_dtrain,num_round)

		xgb.cv(params,x_dtrain,num_round,nfold=5,metrics={'auc'},seed=12345,verbose_eval=True)
		return model,x_dtrain,x_dtest
			


	def mnbayes(self,x_train,y_train):
		model = MultinomialNB()
		model.fit(x_train,y_train)		
		return model

	def svm(self,x_train ,y_train,params=None):		
		if params == None:
			params={'random_state':0,
			'C':10,'loss':'hinge'}
		svm = LinearSVC()
		svm.set_params(**params)        
		#for svm gamma and c need to be learnt using grid search		 
		svm.fit(x_train,y_train)		
		return svm		

	def randforest(self,x_train,y_train,params=None):
		if params == None:
			params={'criterion':'entropy','max_depth':20,'n_estimators':200,
				'random_state':12345,'n_jobs':2}
		forest= RandomForestClassifier()
		forest.set_params(**params)
		#parameters n_estimators ,max_depth need to be learnt
		forest.fit(x_train,y_train)		
		return forest
		





