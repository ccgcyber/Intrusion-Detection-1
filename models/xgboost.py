import xgboost as xgb
from sklearn.model_selection import GridSearchCV as gcv



def xgboost(x_train,y_train,x_test):
	x_dtrain=xgb.DMatrix(data=x_train,labels=y_train)
	x_dtest=xgb.DMatrix(data=x_test)

	params ={'eta': 1,'max_depth':2 ,
	'objective':'binary:logistic'
	}
	num_round = 4
	model = xgb.train(params,x_dtrain,num_round)
	return model 

#hyperparameter optimization
param_grid = { 'eta': [0.01 0.001 0.001],
'max_depth':[],
'gamma': [],
'lambda':[], 
}
grid_search_xgb =gcv(model,param_grid)



