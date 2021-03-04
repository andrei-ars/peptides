import numpy as np
import pandas as pd
import logging
import pickle
from loaddata import DataBase
from sklearn.neural_network import MLPRegressor #MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def train_model(dataset):
    #database.get_file_info(0, 0)
    #file = database.get_file_info(1, 0)
    #for i in file['indices']:
    #    meta_data = database.get_meta_data(i)
    #    print("i={}: {}".format(i, meta_data))
    X = database.get_X()  # returns ndata as np array
    Y = database.get_Y()
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    #X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50, random_state=1) # stratify for classification only
    print(len(X_train), len(X_test), len(Y_train), len(Y_test))

    regr = MLPRegressor(random_state=1, max_iter=200, verbose=True)
    print("training...")
    regr.fit(X_train, Y_train)
    #proba = regr.predict_proba(X_test[:1])
    #print("proba:", proba)
    score = regr.score(X_test, Y_test)
    print("score:", score)
    prediction = regr.predict(X_test[:10, :])
    print("prediction:", prediction)
    print("true values:", Y_test[:10])
    return regr


def find_labels(model, database):
    meta = dataset.get_meta()
    ndata = dataset.get_X()




if __name__ == "__main__":

    #database = DataBase(path="data")
    #model = train_model(dataset)
    #labels = find_labels(model, database)


    meta = pd.read_csv('data/meta.tsv', sep ='\t', index_col = 'index')
    meta['File'] = meta['File'].astype('int64', copy=False)
    peptides = meta["Peptide"].unique()
    for peptide in peptides:
        print("\npeptide: {}".format(peptide))
        meta1 = meta[meta.Peptide==peptide]
        files = {}
        files['A'] = meta1[(meta1.Peptide==peptide) & (meta1.sample_type=="A")].File.unique()
        files['B'] = meta1[(meta1.Peptide==peptide) & (meta1.sample_type=="B")].File.unique()
        files['A'] = list(files['A'])
        files['B'] = list(files['B'])
        for sample_type in ['A', 'B']:
            for file in files[sample_type]:
                print("{} file: {}".format(sample_type, file))
                #file_meta = meta1[(meta1.Peptide==peptide) & (meta1.File==file)]
                indices = meta1[meta1.File==file].index
                print(list(indices))
                clusters = meta.Cluster.iloc[indices]
                print("clusters: {}".format(list(clusters)))
                cluster_mean = meta.ClusterMean.iloc[indices]
                print("cluster_mean: {}".format(list(cluster_mean)))
        #for file in a_files:
        #    print("B file: {}".format(file))
        #    #file_meta = meta1[(meta1.Peptide==peptide) & (meta1.File==file)]
        #    indices = meta1[meta1.File==file].index
        #    print(list(indices))