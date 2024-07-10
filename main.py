from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle
nltk.download('stopwords')
stop_words = stopwords.words('english')
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
model = pickle.load(open('trained_model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.form['a']
    data = stemming(data)
    data = vectorizer.transform([data])
    pred = model.predict(data)
    return render_template('after.html', data=pred[0])

if (__name__ == "__main__"):
    app.run(debug=True)