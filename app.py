
from flask import Flask,jsonify,request 
from sklearn.externals import joblib
import reformat


app=Flask(__name__)

@app.route('/api',methods=['POST'])
def predict_new():

	json=request.get_json()
	temp_df=reformat_json(json)


	predictions=model.predict(temp_df)


	return jsonify(predictions)


if __name__=='__main__':

	model=pickle.load(open('rf_test.pkl','rb'))
	app.run(port = 2345,debug= True)



