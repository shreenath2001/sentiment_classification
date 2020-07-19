import pandas as pd
import numpy as np

df = pd.read_csv("train.txt")

def con_sen_to_rat(sentiment):
    if(sentiment == "positive"):
        return 1
    elif(sentiment == "negative"):
        return -1
    else:
        return 0

df['sentiment'] = df['sentiment'].apply(con_sen_to_rat)

import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import re
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def message_cleaning(message):
    mes_w_o_pun = re.sub("[^a-zA-z]", " ", message)
    remove_stopwords = [ lemmatizer.lemmatize(word) for word in mes_w_o_pun.split() if word.lower() not in set(stopwords.words("english"))   ]
    return remove_stopwords

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(analyzer = message_cleaning)
df_cv = cv.fit_transform(df['tweet_text'])

X_train = df_cv
y_train = df['sentiment']

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=2,
                     solver='adam',
                     verbose=1,
                     random_state=101,
                     early_stopping = True)
mlp.fit(X_train, y_train)

def vect(text_list):
	return cv.transform(text_list)

import pickle
f = open('MLP_classifier.pickle', 'wb')
pickle.dump(mlp, f)
f.close()