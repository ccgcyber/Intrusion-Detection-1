
from flask import Flask,jsonify,request 
from sklearn.externals import joblib
from reformat import reformat_json
import json
app=Flask(__name__)



@app.route('/hello',methods=['GET'])
def something():
	return ('Hey There')

@app.route('/api/bin',methods=['POST'])
def predict_bin():
	print ('Hello World')
	data =json.dumps((request.json))	
	temp_df=reformat_json(data)
	# print (temp_df)
	predictions=model.predict(temp_df)
	
	return 'predictions={}'.format(predictions[0])

@app.route('/api/mlt',methods=['POST'])
def predict_multi():
	print ('Hello World')
	data =json.dumps((request.json))	
	temp_df=reformat_json(data)
	# print (temp_df)
	predictions=model_multi.predict(temp_df)
	
	return 'predictions={}'.format(predictions[0])


if __name__=='__main__':

	model=joblib.load(open('randomforest.pkl','rb'))
	model_multi=joblib.load(open('multirandforest.pkl','rb'))
	app.run(port = 9999,debug= True)



