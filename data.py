import pandas as pd 
import os 
"""
This script fetches the data from the 
root path and returns x_train,y_train,x_test,y_test. 

pathname is the path to the root directory of the project 
"""
def fetch_data(pathname):
	os.chdir(os.path.join(pathname,'data'))
	df_train = pd.read_csv('session_1_data_train.csv')
	df_test = pd.read_csv('session_1_data_test.csv')
	
	print('Datasets loaded :)' )
	print ('The dimensions of the training dataset is {}'.format(df_train.shape))
	assert df_train.shape,(494021,42)
	labelname = 'label'
	y_train=df_train.loc[:,labelname]
	assert y_train.shape,(494021,)
	x_train=df_train.drop(labelname,axis=1,inplace=False)
	assert x_train.shape,(494021,41)

	y_test = df_test.loc[:,labelname]
	x_test = df_test.drop(labelname,axis=1,inplace=False)

	print('The dimensions of test dataset is {}'.format(x_test.shape))
	assert x_test.shape,(311028,42)

	return x_train,y_train,x_test,y_test
