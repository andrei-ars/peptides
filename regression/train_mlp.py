import numpy as np
import pandas as pd
import logging
import pickle
from loaddata import DataBase
from sklearn.neural_network import MLPRegressor #MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time


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

    regr = MLPRegressor(random_state=1, max_iter=3, verbose=True, hidden_layer_sizes=(200, 50))
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


def select_cluster(sample_type, i, measure_pred, value, num_files, real_values):

    print("select_cluster:")
    print("sample_type={}, i={}".format(sample_type, i))
    #print("measure_pred:", measure_pred)
    print("value:", value)
    print("num_files:", num_files)
    print("real_values:", real_values)
    num_a = num_files['A']
    num_b = num_files['B']

    #measure_a = measure_pred['A']
    #measure_b = measure_pred['B']
    #avg_a = np.mean([np.mean(file['values']) for file in measure_a if file['len']>0])
    #avg_b = np.mean([np.mean(file['values']) for file in measure_b if file['len']>0])

    preds = measure_pred[sample_type][i]['values']
    print("sample_type={}, i={}".format(sample_type, i))
    print("preds: {}".format(preds))

    if i == 0:
        avg = np.mean(preds)
        ind = np.argmin(np.abs(preds - avg))
    else:
        numra = len(real_values['A'])
        numrb = len(real_values['B'])
        avg_a = np.mean(real_values['A'])
        avg_b = np.mean(real_values['B'])
        sum_a = np.sum(real_values['B'])
        sum_b = np.sum(real_values['B'])
        print("avg_a(real)={}".format(avg_a))
        print("avg_b(real)={}".format(avg_b))

        if sample_type == "A":
            ind = np.argmin(np.abs( (sum_a + np.array(preds))/(numra+1) - avg_b - value) )
        elif sample_type == "B":
            ind = np.argmin(np.abs( avg_a - (sum_b + np.array(preds))/(numrb+1) - value) )
        else:
            raise Exception("bad sample_type")

        print("--> ind={}".format(ind))

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
        measure_real = {'A': [], 'B': []}
        num_files = {} # for a given peptide

        for sample_type in ['A', 'B']:
            for file in files[sample_type]:
                print("{} file: {}".format(sample_type, file))
                #file_meta = meta1[(meta1.Peptide==peptide) & (meta1.File==file)]
                meta_file = meta1[meta1.File==file]
                indices = meta_file.index
                print(list(indices))
                value = meta_file.value.unique()[0] # = 1,  0, or -2
                print("|value|:", value)
                clusters = meta.Cluster.iloc[indices]
                print("clusters: {}".format(list(clusters)))
                cluster_mean = list(meta.ClusterMean.iloc[indices])
                print("cluster_mean: {}".format(list(cluster_mean)))
                predictions = model.predict(ndata[indices, :])
                predictions = list(predictions)
                print("predictions: {}".format(predictions))

                measure_pred[sample_type].append({'indices': list(indices), 'values': predictions, 'len': len(indices)})
                measure_real[sample_type].append({'indices': list(indices), 'values': cluster_mean, 'len': len(indices)})

            num_files[sample_type] = len(files[sample_type])
            for _ in range(3 - num_files[sample_type]):
                measure_pred[sample_type].append({'indices': None, 'values': None, 'len': 0})
                measure_real[sample_type].append({'indices': None, 'values': None, 'len': 0})

        num_a = num_files['A']  # real number of files
        num_b = num_files['B']  # real number of files
        full_size = 6

        print("measure_pred:", measure_pred)
        assert len(measure_pred['A']) == full_size // 2
        assert len(measure_pred['B']) == full_size // 2
        # measure_pred may have a form [1 1 x 1 x x]
        print("measure_pred (based on prediction results)")
        real_values = {}
        real_values['A'] = []
        real_values['B'] = []

        for k in range(full_size):
            if k % 2 == 0:
                sample_type = 'A'
            else:
                sample_type = 'B'
            i = k // 2

            if i >= num_files[sample_type]:
                continue

            ind = select_cluster(sample_type, i, measure_pred, value, num_files, real_values)
            #ind = select_cluster(sample_type, i, measure_real, value, num_files, real_values)
            real_value = measure_real[sample_type][i]['values'][ind]
            real_values[sample_type].append(real_value)
            index = measure_real[sample_type][i]['indices'][ind]
            selected_indices_total_list.append(index)
            print('{}: i={}, ind={}, index={}'.format(sample_type, i, ind, index))
            print('real_value (added):{}'.format(real_value))
            print('measure_real (all_values):{}'.format(measure_real[sample_type][i]['values']))

        """
        for i in range(num_a):
            ind = select_cluster(sample_type, i, measure_pred, value, num_files, real_values)
            real_value = measure_real['A'][i]['values'][ind]
            real_values['A'].append(real_value)
            index = measure_real['A'][i]['indices'][ind]
            selected_indices_total_list.append(index)

        for i in range(num_b):
            ind = select_cluster('B', i, measure_pred, value, num_files, real_values)
            values = measure_real['B'][i]['values']
            real_value = values[ind]
            real_values['B'].append(real_value)
            index = measure_real['B'][i]['indices'][ind]
            selected_indices_total_list.append(index)
            print('B: i={}, ind={}, index={}'.format(i, ind, index))
            print('real_value (added):{}'.format(real_value))
            print('measure_real (all_values):{}'.format(measure_real['B'][i]['values']))
        """

        mean_a = np.mean(real_values['A'])
        mean_b = np.mean(real_values['B'])
        diff = mean_a - mean_b
        print("mean_a={:.5f}, mean_b={:.5f}".format(mean_a, mean_b))
        print("value={}, diff={:.5f}, error={:.5f} ******\n".format(value, diff, abs(value-diff)))
        #time.sleep(1)

        """
        ind = select_clusters(measure_pred)
        ind_a = ind[:3]  # the local index of measure for a given file
        ind_b = ind[3:]
        #print("ind_a:", ind_a)
        #print("ind_b:", ind_b)

        selected_indices_a = [measure_real['A'][i]['indices'][ind_a[i]] for i in range(num_a)]
        selected_indices_b = [measure_real['B'][i]['indices'][ind_b[i]] for i in range(num_b)]
        selected_indices_total_list += selected_indices_a + selected_indices_b

        real_values_a = [measure_real['A'][i]['values'][ind_a[i]] for i in range(num_a)]
        real_values_b = [measure_real['B'][i]['values'][ind_b[i]] for i in range(num_b)]
        diff = np.mean(real_values_a) - np.mean(real_values_b)
        error = abs(diff - real_value)
        print("selected_indices_a:", selected_indices_a)
        print("selected_indices_b:", selected_indices_b)
        print("real_values_a (selected):", real_values_a)
        print("real_values_b (selected):", real_values_b)
        print("real_diff:", diff)
        print("error:", error)
        print("measure_real (in case if we know the real cluster_mean values)")
        ind = select_clusters(measure_real)
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