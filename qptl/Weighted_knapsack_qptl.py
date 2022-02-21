import os

from qptl.melding_knapsack import *

import time, datetime
import logging

def weighted_knapsack_qptl(train_set, test_set, n_iter=10, capacities=[30], file_name_suffix='empty',
                           dest_folder="Tests/icon/Easy/kfolds/spo/"):
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='../weighted_QPTL.log', level=logging.INFO, format=formatter)


    dir_path = os.path.dirname(os.path.abspath(__file__))

    filename = dest_folder + file_name_suffix
    filename = os.path.join(dir_path, filename)

    X_1gtrain = train_set.get('X')
    y_train = train_set.get('Y')
    X_1gtest = test_set.get('X')
    y_test = test_set.get('Y')

    benchmarks_weights_train = train_set.get('benchmarks_weights')

    X_1gvalidation = X_1gtrain[0:2880, :]
    y_validation = y_train[0:2880]

    y_train = y_train[2880:]
    X_1gtrain = X_1gtrain[2880:, :]

    weights = benchmarks_weights_train[0]

    H_list = [{'optimizer': optim.Adam, 'lr': 1e-4, 'tau': 1000},{'optimizer': optim.Adam, 'tau': 1000}, {'optimizer': optim.Adam, 'tau': 30000}]

    for r in range(n_iter):
        for capa in capacities:
            for h in H_list:
                print("N : %s capacity:%d Time:%s \n" % (str(h), capa, datetime.datetime.now()))
                clf = qptl(capa, weights, epochs=6, validation=True, verbose=True, **h)
                start = time.time()
                pdf = clf.fit(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test)
                end = time.time()
                pdf['capacity'] = capa
                pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
                pdf['total_time'] = end - start
                print(pdf['total_time'])
                # print(pdf['test'])
                filename_full = filename + '_c' + str(capa) + '.csv'
            with open(filename_full, 'a') as f:
            	pdf.to_csv(f, mode='a', header=f.tell() == 0, index=False)
            del pdf

# with open(file_name_full, 'a+') as f:
# 	pdf.to_csv(f, mode='a', header=f.tell() == 0, index=False)

