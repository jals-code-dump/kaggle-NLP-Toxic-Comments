# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 09:27:18 2022

@author: alexc
"""
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# load data
data = pd.read_csv('data_files/scores.csv', index_col=0)
X = data.drop(labels='classification', axis=1)
y = data['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=42)

# fit all models
clf = LazyRegressor(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)