#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 21:17:08 2021

@author: jalaluddin
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv('spam.csv', encoding = 'latin-1')

print(df.columns)

# Feature & Label
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace = True)
df['label'] = df['class'].map({'ham':0, 'spam':1})
print(df.head())

X = df['message']
y = df['label']

cv = CountVectorizer()

# Fit and transform the data
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, 
                                                    random_state = 42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

pickle.dump(cv, open('transform.pkl', 'wb'))
pickle.dump(clf, open('NB_spam_model.pkl', 'wb'))

# ALternative ways of saving the models
#import joblib
#joblib.dump(clf, 'NB_spam_model.pkl')
#joblib.dump(cv, 'transform.pkl')
#file_name = open('NB_spam_model.pkl', 'rb')
#joblib.load(file_name)
