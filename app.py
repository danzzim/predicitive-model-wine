from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('wine_model')
cols = ['fixed acidity','volatile acidity' , 'citric acid' , 'residual sugar'
        , 'chlorides' ,'free sulfur dioxide', 'total sulfur dioxide',
        'density',  'pH', 'sulphates',  'alcohol']

# render default webpage
@app.route('/')
def home():
    return render_template("home.html")

# when the post method detect, predict value
@app.route('/predict' , methods = ['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data= data_unseen, round = 3 )
    prediction = int(prediction.Label[0])
    return render_template('home.html', pred = 'Expected Wine Quality will be {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)

