import numpy as np
import pandas as pd
import logging
import pickle


if __name__ == "__main__":


    data = pd.read_csv('data/data.tsv', sep ='\t', index_col = 'index')
    vdata = data.values # (821654, 32)  this creates a copy of data
    
    print("Replace nan and inf by median value")
    column_median = {
        8:  0.44,
        9:  1.00,
        10: 1.32,
        11: 1.63,
        16: 0.99,
        17: 1.01,
        18: 1.02,
        19: 1.03,
        20: 0.67,
        21: 0.79,
        30: 1.29,
        }
    th = 1000
    for j in column_median:
        print("j = {}".format(j))
        m = column_median[j]
        f = lambda x: m if np.isnan(x) or np.isinf(x) or np.abs(x) > th else x
        vf = np.vectorize(f)
        vdata[:,j] = vf(vdata[:,j])

    ndata = vdata.copy() # normalized data
    for j in range(vdata.shape[1]):
        #print("processing {}-th column".format(j))
        m = vdata[:,j].mean()
        std = vdata[:,j].std()
        print("j={}, m={}, std={}".format(j, m, std))
        #if np.isnan(m) or np.isnan(std):
        #    raise Exception("Bad value")

        if std == 0:
            func = lambda x: x - m
        else:
            func = lambda x: (x - m) / std
        ndata[:,j] = np.apply_along_axis(func, 0, vdata[:,j])

    print("Normalized data:\n", ndata)

    outpath = "data/ndata.pickle"
    with open(outpath, 'wb') as fp:
        pickle.dump(ndata, fp)
        print("Data has been saved in {}".format(outpath))
    
#-------------

for j in range(vdata.shape[1]):
    print("processing {}-th column".format(j))
    m = vdata[:,j].mean()
    std = vdata[:,j].std()
    func = lambda x: (x - m) / std
    ndata[:,j] = np.apply_along_axis(func, 0, vdata[:,j])