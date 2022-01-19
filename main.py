"""
Created on Sat Jan 15 18:09:14 2022

@author: alexc
"""

import pandas as pd
import nltk
import re
import enchant
from nltk.sentiment import SentimentIntensityAnalyzer

# load datasets
val_df = pd.read_csv('data_files/validation_data.csv')
a_file = open("data_files/bad-words.txt", "r")
bad_words = [(line.strip()) for line in a_file]
a_file.close()
text_string = """And you removed it! You numbskull! I don't care what you 
                 say anymore, this is my life! Go ahead with your own life, 
                 leave me alone!"""

# setup pyenchant
enc_dict = enchant.DictWithPWL("en_GB", "data_files/bad-words.txt")

# setup sentiment analyzer
sia = SentimentIntensityAnalyzer()

def ratio(val1, val2):
    try:
        ratio = val1 / val2
    except:
        ratio = 0
    return ratio


def score(text_string, classifier):
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
                    if not enc_dict.check(word)])
    scores.append(ratio(spelling, total_words))
    # print("Spelling Errors:", spelling)

    # positivity
    scores.append(sia.polarity_scores(text_string).get('compound'))

    # add manual classifier
    scores.append(classifier)
    return scores


results = []
val_df['more_toxic'].apply(lambda x: results.append(score(x, 1.0)))
val_df['less_toxic'].apply(lambda x: results.append(score(x, 0.0)))
col_names = ['capital_ratio', 'punctuation_ratio', 'offensive_ratio',
             'spelling_errors', 'positivity', 'classification']
res_df = pd.DataFrame(data=results, columns=col_names)
res_df.to_csv('data_files\scores.csv')
