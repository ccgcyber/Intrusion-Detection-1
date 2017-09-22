import json 
import pandas as pd 
"""
Test json needs to be changed as the feature space is the same as the 
training data.

converts the test json into a modified test dataframe.

"""

def reformat_json(json):
	

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

		dict_temp=json.loads(json)
		df_temp=pd.DataFrame(dict_temp,index=[0])

		df_temp['service']=df_temp['service'].apply(lambda x: bin_service(x))
		df_temp=pd.get_dummies(df_temp,columns=['service'],prefix='service',drop_first=True)
		
		df_temp['flag']=df_temp['flag'].apply(lambda x: bin_flag(x))
		df_temp=pd.get_dummies(df_temp,columns=['flag'],prefix='flag',drop_first=True)

		features_to_be_removed=['num_root','su_attempted','num_outbound_cmds','is_host_login']
		df_temp.drop(features_to_be_removed,axis=1,inplace=True)


		return df_temp 









