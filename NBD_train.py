from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Any

import codecs
import json
# import PyQt5
# from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
# import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import redirect

# Import the following libraries - Pandas, Numpy, Goose3, re, nltk, sklearn
# Importing required libraries
import numpy as np
# !pip install pandas
import pandas as pd
# !pip install goose3
from goose3 import Goose
import pickle

import re
import nltk


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('wordnet')
nltk.download('omw-1.4')

# !pip install sklearn

# Function for processing the texts


def preprocessing_text(text, stem=False, lemma=True):
    # clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    # Tokenization (convert from string to list)
    processed_text = text.split()

    # Stemming (remove -ing, -ly, ...)
    if stem == True:
        ps = nltk.stem.porter.PorterStemmer()
        processed_text = [ps.stem(word) for word in processed_text]

    # Lemmatisation (convert the word into root word)
    if lemma == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        processed_text = [lem.lemmatize(word) for word in processed_text]

    # back to string from list
    text = " ".join(processed_text)
    return text


# Importing dataset
path = 'Political_Data.csv'
# Dataset can be stored locally and the provide the path in the "path" variable
df = pd.read_csv(path)

# Dataset link - https://docs.google.com/spreadsheets/d/1vt9lmAd8QyjoFZjFg6ID0TguubzX9BLU/edit#gid=63028571


# displaying dataset
df

X_origin = df['Article']
Y = df['Annotation']

df["text_clean"] = df['Article'].apply(lambda x:
                                       preprocessing_text(x, stem=False, lemma=True))
df['Y_cln'] = df['Annotation'].apply(lambda x:
                                     preprocessing_text(x, stem=False, lemma=False))

X_clean = df['text_clean']
Y_clean = df['Y_cln']

# Encoding political parties to a number

le = LabelEncoder()
Y_clean = le.fit_transform(Y_clean)
print(Y_clean)
# 1 - bjp
# 2 - cong
# 3 - None
# 0 - AAP

# Splitting dataset into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_clean, Y_clean, random_state=4, test_size=0.2)
# print(X_train)
# print(Y_train)
# X_train = X_train.reset_index()
# X_test = X_test.reset_index()
# print(X_train)

# Using Tfidf Vectorizer to form word vectors

v = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
Trainvectf = v.fit_transform(X_train)
testvectf = v.transform(X_test)

# Importing gradient Boosting Classifier

# Fitting the dataset and training the model
gb = GradientBoostingClassifier(
    n_estimators=160, learning_rate=0.05, random_state=100, max_depth=3)
gb.fit(Trainvectf, Y_train)

# Prediction using test set
test_prd_gbgb = gb.predict(testvectf)

# Showing accuracy and other metrics of the model on test set
print(classification_report(Y_test, test_prd_gbgb))  # best with Tdidf

# Save the vectorizer
pickle.dump(v, open("./model/vectorizer.sav", 'wb'))

# Save the model
pickle.dump(gb, open('./model/NBD_model.sav', 'wb'))


# Put the article link here to get it's bias results here

app = Flask(__name__, template_folder="C:\\Users\\akash")
hihi = None
hihi = 0
print('Yo?')


@app.route('/', methods=['POST'])
def sm():
    # Put article link in front of url in '' single quotes
    if request.method == 'POST':
        h = request.json
        print("hr", h)
        url = h
        # url = 'PUT YOUR LINK HERE'
        g = Goose()
        article = g.extract(url=url)
        trying = article.cleaned_text
        better = preprocessing_text(trying, stem=False, lemma=True)
        b = pd.Series(better)
        hh = v.transform(b)
        # predicting the bias of the article towards or against a particular party
        oc = gb.predict(hh)
        # 1 - bjp
        # 2 - cong
        # 3 - None
        # 0 - AAP
        # NOTE: Here "bias towards" means that the article maybe biased towards or against a particular political party
        dd = {}
        k = 0
        if oc == 0:
            print("News article is biased towards AAP")
            k = 0
            # return 'News article is biased towards AAP'
        elif oc == 1:
            print("News article is biased towards BJP")
            k = 1
            # return '<h1>News article is biased towards BJP</h1>'
            # dd = {"News article is biased towards BJP"}
        elif oc == 2:
            k = 2
            print("News article is biased towards Congress")
            # return 'News article is biased towards Congress'
            # dd = {"News article is biased towards Congress"}
        else:
            print("News article is unbiased")
            k = 4
            # return 'News article is unbiased'
            # dd = {"News article is unbiased"}
        dd['ans'] = k
        global hihi
        hihi = k
        print(hihi)
        print("POST TRYNA HATE ON US")
        print(dd)
        return redirect(url_for('dt'))
        # return '<h1>The predicted bias is <u></u></h1>'
        return dd
    else:
        print("BHAKKA")
        hihi = 0
        return redirect(url_for('dt', ff=hihi))


@app.route('/hi', methods=['GET', 'POST'])
def dt():
    global b
    d = {}
    print(hihi)
    d['ans'] = hihi
    return d


if __name__ == '__main__':
    app.run()
