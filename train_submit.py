# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 10:10:45 2022

@author: alexc
"""

import pandas as pd
from xgboost import XGBRegressor
import nltk
import re
import enchant
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

training_data = pd.read_csv('data_files/scores.csv', index_col=0)
X = np.array(training_data.drop(labels='classification', axis=1))
y = np.array(training_data.classification)

model = XGBRegressor()
model = model.fit(X, y)

###

# process submission sample
a_file = open("data_files/bad-words.txt", "r")
bad_words = [(line.strip()) for line in a_file]
a_file.close()
sample = pd.read_csv('data_files/comments_to_score.csv', index_col=0)

# setup dictionary
enc_dict = nltk.corpus.words.words()
for word in bad_words:
    enc_dict.append("word")

# setup sentiment analyzer
sia = SentimentIntensityAnalyzer()


def ratio(val1, val2):
    try:
        ratio = val1 / val2
    except:
        ratio = 0
    return ratio


def score(text_string):
    # score holder
    scores = []

    # capital ratio
    capitals = len(re.compile('[A-Z]').findall(text_string))
    lowers = len(re.compile('[a-z]').findall(text_string))
    scores.append(ratio(capitals, lowers))
    # print("Capitals and lowers:", capitals, lowers)

    # punctuation ratio
    punct = len(re.compile('[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]')  # noqa
                .findall(text_string))
    non_punct = len(re.compile('[\d\w]').findall(text_string))  # noqa
    scores.append(ratio(punct, non_punct))
    # print("Punctuation and characters:", punct, non_punct)

    # clean text string
    # extract characters and reduce to lower case
    word_bag = re.sub('[^A-Za-z]', ' ', text_string).lower()
    # tokenise words
    word_bag = nltk.tokenize.word_tokenize(word_bag)
    total_words = len(word_bag)

    # offensive ratio
    # count amount of offensive words
    offensive = len([word for word in word_bag if word in bad_words])
    scores.append(ratio(offensive, total_words))
    # print("Offensive Words:", offensive)

    # spelling errors ratio
    spelling = len([word for word in word_bag
                    if word not in enc_dict])
    scores.append(ratio(spelling, total_words))
    # print("Spelling Errors:", spelling)

    # positivity
    scores.append(sia.polarity_scores(text_string).get('compound'))
    scores = np.array(scores).reshape(1, -1)
    result = model.predict(scores)[0]
    print(scores)
    return result

sample['score'] = sample.text.apply(lambda x: score(x))
sample.drop(labels='text', inplace=True, axis=1)
sample.to_csv('data_files/submission.csv')
