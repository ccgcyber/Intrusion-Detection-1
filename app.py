
from flask import Flask,jsonify,request 
from sklearn.externals import joblib
from reformat import reformat_json
import json
app=Flask(__name__)



@app.route('/hello',methods=['GET'])
def something():
	return ('Hey There')

@app.route('/api',methods=['POST'])
def predict_new():
	print ('Hello World')
	data =json.dumps((request.json))

	
	
	temp_df=reformat_json(data)
	# print (temp_df)



	predictions=model.predict(temp_df)
	# print('predictions ={}'.format(predictions[0]))
	


	return 'predictions={}'.format(predictions[0])

if __name__=='__main__':

	model=joblib.load(open('randomforest.pkl','rb'))
	app.run(port = 9999,debug= True)



