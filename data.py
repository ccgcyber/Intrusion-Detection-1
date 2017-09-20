import pandas as pd 
import numpy as np
import os
"""
This script fetches the data from the 
root path and returns x_train,y_train,x_test,y_test. 

pathname is the path to the root directory of the project 
"""
def fetch_data(pathname,remove_duplicates=False):
	os.chdir(os.path.join(pathname,'data'))
	df_train = pd.read_csv('session_1_data_train.csv')
	df_test = pd.read_csv('session_1_data_test.csv')
	
	assert df_train.shape,(494021,42)
	labelname = 'label'
	y_train=df_train.loc[:,labelname]
	assert y_train.shape,(494021,)
	x_train=df_train.drop(labelname,axis=1,inplace=False)
	assert x_train.shape,(494021,41)

	y_test = df_test.loc[:,labelname]
	x_test = df_test.drop(labelname,axis=1,inplace=False)
	assert x_test.shape,(311028,42)

	if remove_duplicates == True:
		temp_train=x_train.copy()
		x_train=x_train[pd.DataFrame(~np.array( temp_train.duplicated()))[0]]
		assert x_train.shape,(145583,41)
		y_train=y_train[pd.DataFrame(~np.array(temp_train.duplicated()))[0]]
		assert y_train.shape ,(145583,)

		print('Datasets loaded :)' )
		print ('The dimensions of the training dataset is {}'.format(x_train.shape))
		print('The dimensions of test dataset is {}'.format(x_test.shape))

		return x_train,y_train,x_test,y_test

	else:
		print('Datasets loaded :)' )
		print ('The dimensions of the training dataset is {}'.format(x_train.shape))
		print('The dimensions of test dataset is {}'.format(x_test.shape))
		return x_train,y_train,x_test,y_test


def feature_engineering(x_train): 
	# check always that x_train.shape[0] does not change throughout
	# feature engineering on the training data

	def bin_service(x):
		if x in {'http','private','smtp','domain_u','other','ftp_data'}:
			return x
		else:
			return 'others'

	def bin_flag(x):
		if x in {'SF','S0','REJ'}:
			return x
		else:
			return 'others'


	x_train['service']=x_train['service'].apply(lambda x: bin_service(x))
	x_train=pd.get_dummies(x_train,columns=['service'],prefix='service',drop_first=True)
	x_train['flag']=x_train['flag'].apply(lambda x: bin_flag(x))
	x_train=pd.get_dummies(x_train,columns=['flag'],prefix='flag',drop_first=True)


	return x_train 



