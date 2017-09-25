import json 
import pandas as pd 
from data import feature_engineering
"""
Test json needs to be changed as the feature space is the same as the 
training data+
converts the test json into a modified test dataframe.

"""

def reformat_json(data):
	df_sample=pd.read_csv('df_sample.csv')
	df_sample=df_sample.drop(['Unnamed: 0'],axis=1,inplace=False)
	test_df=pd.DataFrame(json.loads(data))
	# print (test_df)

	df_test=pd.concat([test_df,df_sample])
	df_test.reset_index()
	#print (df_sample)

	# df_test.drop(['index','level_0'],axis=1,inplace=False)

	df_test=feature_engineering(df_test,do_normalization=False)
	# df_test=df_test.drop(['Unnamed: 0'],axis=1,inplace=False)

	return df_test











