# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Fri Mar 12 21:17:08 2021

@author: jalaluddin

"""
from flask import Flask, request, render_template, url_for
import pickle

# Load the models
clf = pickle.load(open('NB_spam_model.pkl', 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug = 'True')