import pandas as pd
import numpy as np

train_df = pd.read_csv("train.txt")

def con_sen_to_rat(sentiment):
    if(sentiment == "positive"):
        return 1
    elif(sentiment == "negative"):
        return -1
    else:
        return 0

train_df['sentiment'] = train_df['sentiment'].apply(con_sen_to_rat)

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
df_cv = cv.fit_transform(train_df['tweet_text'])

X_train = df_cv[:21465]
y_train = train_df['sentiment'][:21465]

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

def vect(text_list):
	return cv.transform(text_list)

import pickle
f = open('NB_classifier.pickle', 'wb')
pickle.dump(NB_classifier, f)
f.close()