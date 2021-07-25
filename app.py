import flask
from flask import Flask,render_template,url_for,request
import numpy as np
import tensorflow
#import re
import keras
from tensorflow import keras
from keras import models
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)
clf = load_model('model_aann.h5')

def preprocess(text):
    #space = re.compile('[/(){}\[\]\|@,;]')
    #symbols = re.compile('[^0-9a-z #+_]')
    #STOPWORDS = set(stopwords.words('english'))


    text = text.lower()  # lowercase text
    #text = space.sub(' ',text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    #text = symbols.sub('',text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
        #    text = re.sub(r'\W+', '', text)
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text


    from keras.preprocessing.text import Tokenizer
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 3000
    # This is fixed.
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    ### Tokenize preprocessed data
    # Truncate and pad the input sequences so that they are all in the same length for modeling.

    from keras.preprocessing.sequence import pad_sequences
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    return X

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
    data = request.form['message']
   
    data = preprocess(data)
    '''
    my_prediction = clf.predict(data)
    labels = ['Stage_1', 'Stage_2', 'Stage_3', 'Stage_4','Stage_5']
    result = np.argmax(my_prediction)
    result = labels[result]
    '''
    pred = clf.predict(data)
    labels = np.array(['Stage_1', 'Stage_2', 'Stage_3', 'Stage_4', 'Stage_5'])
    labels.sort()
    results = np.argmax(pred,axis=-1)
    predicted_label = labels[results][1]

    
    labels = ['Stage_1', 'Stage_2', 'Stage_3', 'Stage_4', 'Stage_5']



    #output = round(prediction[0], 2)
    return render_template('home2.html', prediction_text="The Conflict Stage is {}".format(predicted_label))

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
