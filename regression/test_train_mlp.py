import numpy as np
import pandas as pd
import logging
from train_mlp import select_cluster


if __name__ == "__main__":

    sample_type = 'A'
    i=1
    measure_pred = {'A': 
            [{'indices': [0, 1, 2, 3], 'values': [11.891938487170371, 14.535763188178562, 13.114688280462293, 12.37019355638733], 'len': 4}, 
            {'indices': [6, 7, 8, 9], 'values': [12.921119947281145, 12.100478381110072, 12.994374970978969, 11.189299373108579], 'len': 4}, 
            {'indices': [13, 14, 15, 16, 17], 'values': [13.698045711991082, 13.123181622201791, 13.319816512523964, 12.545678639208386, 12.205612289081772], 'len': 5}], 
            'B': [{'indices': [4, 5], 'values': [11.888151783126244, 12.216035949747528], 'len': 2},
            {'indices': [10, 11, 12], 'values': [12.08090304747445, 13.604577086835869, 12.75023769638919], 'len': 3}, 
            {'indices': [18, 19, 20], 'values': [12.424533915710427, 12.211994339095556, 11.835946063253896], 'len': 3}]}
    value = 1
    num_files = {'A': 3, 'B': 3}
    true_values = {'A': [13.144618152755024], 'B': []}

    ind = select_cluster(sample_type, i, measure_pred, value, num_files, 
                                                            true_values)
    print('ind:', ind)
    true_value = measure_true['A'][i]['values'][ind]
    print('true_value:', true_value)