from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

def k_fold_crossval(model,x_train,y_train,K=None):
	if K==None:
		K=3
	f1score=[] 
	counter=1  
	kf = KFold(n_splits = K, shuffle = True)
	for result in kf.split(y_train):
		X_train,Y_train= x_train.iloc[result[0]],y_train.iloc[result[0]]
		X_test,Y_test = x_train.iloc[result[1]],y_train.iloc[result[1]]
		   
		model.fit(X_train,Y_train)    
		Y_pred=model.predict(X_test)
		f1= f1_score(Y_test, Y_pred)
		print('f1 score for {} iteration................. {:.4f}'.format(counter,f1))
		f1score.append(f1)
		counter+=1
	return np.mean(f1score)
	



            
	




