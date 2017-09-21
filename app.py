import numpy as np 
import pickle 
from flask import Flask,jsonify,request 
from app import test_format 


app=Flask(__name__)
model=pickle.load(model,)

@app.route('/api',methods=['POST'])
def predict_new():
	json=request.get_json()
	temp_df=reformat_json(json)


	predictions=model.predict(temp_df)


	return jsonify(predictions)


if __name__=='__main__':
	app.run(port = 9999,debug= True)