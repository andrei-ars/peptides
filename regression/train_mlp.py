import numpy as np
import pandas as pd
import logging
import pickle
from loaddata import DataBase
from sklearn.neural_network import MLPRegressor #MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    database = DataBase(path="data")
    #database.get_file_info(0, 0)
    #file = database.get_file_info(1, 0)
    #for i in file['indices']:
    #    meta_data = database.get_meta_data(i)
    #    print("i={}: {}".format(i, meta_data))
    X = database.get_X()
    y = database.get_Y()
    print("X.shape:", X.shape)
    print("Y.shape:", y.shape)

    #X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) # stratify for classification only
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    regr = MLPRegressor(random_state=1, max_iter=300, verbose=True)
    print("training...")
    regr.fit(X_train, y_train)
    #proba = regr.predict_proba(X_test[:1])
    #print("proba:", proba)
    score = regr.score(X_test, y_test)
    print("score:", score)
    prediction = regr.predict(X_test[:10, :])
    print("prediction:", prediction)
    print("true values:", y_test[:10])
