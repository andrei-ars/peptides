import numpy as np
import pandas as pd
import logging
import pickle
from loaddata import DataBase
from sklearn.neural_network import MLPRegressor #MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def train_model(database):
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

    regr = MLPRegressor(random_state=1, max_iter=15, verbose=True)
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

"""
def select_clusters(measure, value):

    measure_a = measure['A']
    measure_b = measure['B']
    avg_a = np.mean([np.mean(file['values']) for file in measure_a if file['len']>0])
    avg_b = np.mean([np.mean(file['values']) for file in measure_b if file['len']>0])
    diff = avg_a - avg_b
    print("avg_a:", avg_a)
    print("avg_b:", avg_b)
    print("diff:", diff)
    # Find value [ 1,  0, -2]
    if diff > 0.5:
        value = 1
    elif diff > -0.7:
        value = 0
    else:
        value = -2
    print("calculated value:", value)

    len_a0 = measure_a[0]['len']
    len_a1 = measure_a[1]['len']
    len_a2 = measure_a[2]['len']
    len_b0 = measure_b[0]['len']
    len_b1 = measure_b[1]['len']
    len_b2 = measure_b[2]['len']
    
    if len_a1 == 0:
        num_a = 1
        len_a1 = len_a2 = 1
    elif len_a2 == 0:
        num_a = 2
        len_a2 = 1
    else:
        num_a = 3

    if len_b1 == 0:
        num_b = 1
        len_b1 = len_b2 = 1
    elif len_b2 == 0:
        num_b = 2
        len_b2 = 1
    else:
        num_b = 3

    diff_array = np.zeros((len_a0, len_a1, len_a2, len_b0, len_b1, len_b2), dtype=float)

    for ia0 in range(len_a0):
        for ia1 in range(len_a1):
            for ia2 in range(len_a2):
                for ib0 in range(len_b0):
                    for ib1 in range(len_b1):
                        for ib2 in range(len_b2):

                            if num_a == 3:
                                a0 = measure_a[0]['values'][ia0]
                                a1 = measure_a[1]['values'][ia1]
                                a2 = measure_a[2]['values'][ia2]
                                a_avg = np.mean([a0, a1, a2])
                            elif num_a == 2:
                                a0 = measure_a[0]['values'][ia0]
                                a1 = measure_a[1]['values'][ia1]
                                a_avg = np.mean([a0, a1])
                            else:
                                a0 = measure_a[0]['values'][ia0]
                                a_avg = a0

                            if num_b == 3:
                                b0 = measure_b[0]['values'][ib0]
                                b1 = measure_b[1]['values'][ib1]
                                b2 = measure_b[2]['values'][ib2]
                                b_avg = np.mean([b0, b1, b2])
                            elif num_b == 2:
                                b0 = measure_b[0]['values'][ib0]
                                b1 = measure_b[1]['values'][ib1]
                                b_avg = np.mean([b0, b1])
                            else:
                                b0 = measure_b[0]['values'][ib0]
                                b_avg = b0

                            diff = a_avg - b_avg
                            diff_array[ia0,ia1,ia2,ib0,ib1,ib2] = diff
                            #print(ia0,ia1,ia2,ib0,ib1,ib2,diff)

    #print("diff_array:", diff_array)
    ind = np.unravel_index(np.argmin(np.abs(diff_array - value), axis=None), diff_array.shape)
    closed_value = diff_array[ind]
    print("closed_value = {} with ind = {}".format(closed_value, ind))
    return ind
"""


def select_cluster(sample_type, i, measure_pred, value, num_files, true_values):

    print("select_cluster:")
    print("sample_type={}, i={}".format(sample_type, i))
    print("measure_pred:", measure_pred)
    print("value:", value)
    print("num_files:", num_files)
    print("true_values:", true_values)

    measure_a = measure_pred['A']
    measure_b = measure_pred['B']
    avg_a = np.mean([np.mean(file['values']) for file in measure_a if file['len']>0])
    avg_b = np.mean([np.mean(file['values']) for file in measure_b if file['len']>0])

    len_a0 = measure_a[0]['len']
    len_a1 = measure_a[1]['len']
    len_a2 = measure_a[2]['len']
    len_b0 = measure_b[0]['len']
    len_b1 = measure_b[1]['len']
    len_b2 = measure_b[2]['len']
    
    if len_a1 == 0:
        #num_a = 1
        len_a1 = len_a2 = 1
    elif len_a2 == 0:
        #num_a = 2
        len_a2 = 1
    else:
        pass
        #num_a = 3

    if len_b1 == 0:
        num_b = 1
        #len_b1 = len_b2 = 1
    elif len_b2 == 0:
        #num_b = 2
        len_b2 = 1
    else:
        pass
        #num_b = 3

    if sample_type == "A":
        if i == 0:
            avg = np.mean(measure_a[i]['values'])
            ind = np.argmin(np.abs(measure_a[i]['values'] - avg))
            return ind
        else:
            avg = np.mean(true_values['A'])
            ind = np.argmin(np.abs(measure_a[i]['values'] - avg))
            return ind

    elif sample_type == "B":
        preds = measure_b[i]['values']
        avg_a = np.mean(true_values['A'])
        if i == 0:
            ind = np.argmin(np.abs(avg_a - preds - value))
        else:
            avg_b = np.mean(true_values['B'])
            ind = np.argmin(np.abs(avg_a - 0.5*(avg_b + preds) - value))
        return ind


    #print("diff_array:", diff_array)
    #ind = np.unravel_index(np.argmin(np.abs(diff_array - value), axis=None), diff_array.shape)
    #closed_value = diff_array[ind]
    #print("closed_value = {} with ind = {}".format(closed_value, ind))



def find_labels(model, database):
    meta = database.get_meta()
    ndata = database.get_X()
    #meta = pd.read_csv('data/meta.tsv', sep ='\t', index_col = 'index')
    #meta['File'] = meta['File'].astype('int64', copy=False)

    selected_indices_total_list = []

    peptides = meta["Peptide"].unique()
    for peptide in peptides:
        print("\npeptide: {}".format(peptide))
        meta1 = meta[meta.Peptide==peptide]
        files = {}
        files['A'] = meta1[(meta1.Peptide==peptide) & (meta1.sample_type=="A")].File.unique()
        files['B'] = meta1[(meta1.Peptide==peptide) & (meta1.sample_type=="B")].File.unique()
        files['A'] = list(files['A'])
        files['B'] = list(files['B'])
        
        measure_pred = {'A': [], 'B': []}
        measure_true = {'A': [], 'B': []}
        num_files = {} # for a given peptide

        for sample_type in ['A', 'B']:
            for file in files[sample_type]:
                print("{} file: {}".format(sample_type, file))
                #file_meta = meta1[(meta1.Peptide==peptide) & (meta1.File==file)]
                meta_file = meta1[meta1.File==file]
                indices = meta_file.index
                print(list(indices))
                value = meta_file.value.unique()[0] # = 1,  0, or -2
                print("true_value:", value) 
                clusters = meta.Cluster.iloc[indices]
                print("clusters: {}".format(list(clusters)))
                cluster_mean = list(meta.ClusterMean.iloc[indices])
                print("cluster_mean: {}".format(list(cluster_mean)))
                predictions = model.predict(ndata[indices, :])
                predictions = list(predictions)
                print("predictions: {}".format(predictions))

                measure_pred[sample_type].append({'indices': list(indices), 'values': predictions, 'len': len(indices)})
                measure_true[sample_type].append({'indices': list(indices), 'values': cluster_mean, 'len': len(indices)})

            num_files[sample_type] = len(files[sample_type])
            for _ in range(3 - num_files[sample_type]):
                measure_pred[sample_type].append({'indices': None, 'values': None, 'len': 0})
                measure_true[sample_type].append({'indices': None, 'values': None, 'len': 0})

        num_a = num_files['A']  # real number of files
        num_b = num_files['B']  # real number of files
        full_size = 6

        print("measure_pred:", measure_pred)
        assert len(measure_pred['A']) == full_size // 2
        assert len(measure_pred['B']) == full_size // 2
        # measure_pred may have a form [1 1 x 1 x x]
        print("measure_pred (based on prediction results)")
        true_values = {}
        true_values['A'] = []
        true_values['B'] = []

        for i in range(num_a):
            ind = select_cluster('A', i, measure_pred, value, num_files, true_values)
            print('ind:', ind)
            true_value = measure_true['A'][i]['values'][ind]
            true_values['A'].append(true_value)
            selected_indices_total_list.append(ind)

        for i in range(num_b):
            ind = select_cluster('B', i, measure_pred, value, num_files, true_values)
            print('ind:', ind)
            print('i={}'.format(i))
            values = measure_true['B'][i]['values']
            print('values:{}'.format(values))
            true_value = values[ind]
            true_values['B'].append(true_value)
            selected_indices_total_list.append(ind)


        """
        ind = select_clusters(measure_pred)
        ind_a = ind[:3]  # the local index of measure for a given file
        ind_b = ind[3:]
        #print("ind_a:", ind_a)
        #print("ind_b:", ind_b)

        selected_indices_a = [measure_true['A'][i]['indices'][ind_a[i]] for i in range(num_a)]
        selected_indices_b = [measure_true['B'][i]['indices'][ind_b[i]] for i in range(num_b)]
        selected_indices_total_list += selected_indices_a + selected_indices_b

        true_values_a = [measure_true['A'][i]['values'][ind_a[i]] for i in range(num_a)]
        true_values_b = [measure_true['B'][i]['values'][ind_b[i]] for i in range(num_b)]
        diff = np.mean(true_values_a) - np.mean(true_values_b)
        error = abs(diff - true_value)
        print("selected_indices_a:", selected_indices_a)
        print("selected_indices_b:", selected_indices_b)
        print("true_values_a (selected):", true_values_a)
        print("true_values_b (selected):", true_values_b)
        print("real_diff:", diff)
        print("error:", error)
        print("measure_true (in case if we know the real cluster_mean values)")
        ind = select_clusters(measure_true)
        """

    return selected_indices_total_list


if __name__ == "__main__":

    database = DataBase(path="data")
    model = train_model(database)
    selected_indices = find_labels(model, database)

    database.add_label_column(selected_indices)
    #meta['label'] = 0


    #for file in a_files:
    #    print("B file: {}".format(file))
    #    #file_meta = meta1[(meta1.Peptide==peptide) & (meta1.File==file)]
    #    indices = meta1[meta1.File==file].index
    #    print(list(indices))