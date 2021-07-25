import flask
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import pickle

'''
import tensorflow
import keras
from tensorflow import keras
from keras import models
from keras.models import load_model
#from tensorflow.keras.preprocessing.text import Tokenizer
'''

app = Flask(__name__)
clf = load_model('model_aann.h5')
#model = pickle.load(open('randomForestRegressor.pkl','rb'))
#filename = 'model.pkl'
#clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home2.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction[0])
    '''
    message = request.form['message']
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)

    #output = round(prediction[0], 2)
    return render_template('home2.html', prediction_text="The Conflict Stage is {}".format(my_prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = clf.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)