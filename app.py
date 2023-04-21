from flask import Flask, request

from goose3 import Goose

import sklearn
import pandas as pd
import pickle
import nltk
import re

nltk.download('omw-1.4')
nltk.download('wordnet')


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


# Importing vectorizer
vectorizer = pickle.load(open('./model/vectorizer.sav', 'rb'))


# Importing Model
model = pickle.load(open('./model/NBD_model.sav', 'rb'))


# Flask app

app = Flask(__name__)

print('Server Started')

@app.route('/', methods=['GET'])
def home():
    return 'Hello, World!'


@app.route('/article', methods=['POST'])
def article():
    # Put article link in front of url in '' single quotes
    if request.method == 'POST':

        URL = request.json
        print("URL :: ", URL)

        goose = Goose()
        raw_article = goose.extract(url=URL)
        cleaned_article = raw_article.cleaned_text
        article = preprocessing_text(cleaned_article, stem=False, lemma=True)

        article_array = pd.Series(article)
        result = vectorizer.transform(article_array)

        # predicting the bias of the article towards or against a particular party
        predictedVal = model.predict(result)
        # 0 - AAP
        # 1 - bjp
        # 2 - cong
        # 3 - None
        # NOTE: Here "bias towards" means that the article maybe biased towards or against a particular political party
        res = {}
        k = 3
        if predictedVal == 0:
            print("News article is biased towards AAP")
            k = 0
        elif predictedVal == 1:
            print("News article is biased towards BJP")
            k = 1
        elif predictedVal == 2:
            k = 2
            print("News article is biased towards Congress")
        else:
            print("News article is unbiased")
            k = 3

        res['predicted_val'] = k
        return res

    else:
        print("No response from the Model")
        res['predicted_val'] = 4
        return res


if __name__ == '__main__':
    app.run()
