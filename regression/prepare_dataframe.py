import sys
import numpy as np
import pandas as pd
import logging
import pickle


def prepare_dataframe():

    print("Loading data...")
    data = pd.read_csv('data/data.tsv', sep ='\t', index_col = 'index')
    #vdata = data.values # (821654, 32)  this creates a copy of data
    data = data[(data.sample_type == "A") & (data.value == 1)]

    print("Replace nan and inf by median value")
    column_median = {
        'min_rmse_h_ratio':  0.44,
        'median_rmse_h_ratio':  1.00,
        'q3_rmse_h_ratio': 1.32,
        'q4_rmse_h_ratio': 1.63,
        'median_corr': 0.99,
        'q3_corr': 1.01,
        'q4_corr': 1.02,
        'max_corr': 1.03,
        'median_cluster_cross_corr': 0.67,
        'q3_cluster_cross_corr': 0.79,
        'NNd': 1.29,
        }
    th = 1000 # max threshold
    for colomn in column_median:
        print("colomn: {}".format(colomn))
        m = column_median[colomn]
        f = lambda x: m if np.isnan(x) or np.isinf(x) or np.abs(x) > th else x
        #vf = np.vectorize(f)
        data[colomn] = data[colomn].apply(f)

    #ndata = vdata.copy() # normalized data
    print("Normalization")
    for colomn in data.columns:
        #print("processing {}-th column".format(j))
        m = data[colomn].mean()
        std = data[colomn].std()
        print("j={}, m={}, std={}".format(j, m, std))
        #if np.isnan(m) or np.isnan(std):
        #    raise Exception("Bad value")
        if std == 0:
            func = lambda x: x - m
        else:
            func = lambda x: (x - m) / std
        data[colomn] = data[colomn].apply(func)

    print("Normalized data:\n", data)

    outpath = "data/ndataframe.pickle"
    with open(outpath, 'wb') as fp:
        pickle.dump(data, fp)
        print("Data has been saved in {}".format(outpath))


if __name__ == "__main__":
    prepare_dataframe()