import os

from IntOpt.Interior.intopt_energy_mlp import intopt_energy
from dnl.PredictPlusOptimize import PredictPlusOptModel
from qptl.melding_knapsack import *

import multiprocessing as mp
import time, datetime
import logging

def weighted_knapsack_intopt(train_set, test_set, n_iter=10, capacity=1, opt_params=None, file_name_suffix='empty',
                        dest_folder="Tests/experimental/", time_limit=12000, epoch_limit=1):
    # dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # file_path = os.path.join(dir_path, 'data', "prices2013.dat")
    #
    # (X_1gtrain, y_train, X_1gtest, y_test) = get_energy(file_path)
    # X_1gvalidation = X_1gtest[0:2880, :]
    # y_validation = y_test[0:2880]
    # y_test = y_test[2880:]
    # X_1gtest = X_1gtest[2880:, :]
    # weights = [[1 for i in range(48)]]
    # weights = np.array(weights)
    # X_1gtrain = X_1gtrain[:, 1:]
    # X_1gvalidation = X_1gvalidation[:, 1:]
    # X_1gtest = X_1gtest[:, 1:]
    #
    # param_path = os.path.join(dir_path, "data/icon_instances/easy", "instance34.txt")
    # param = opt_params

    #sssssssssssssssssssssss

    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # file = "data/icon_instances/easy/instance" + str(capacities) + ".txt"
    # file = os.path.join(dir_path, file)

    filename = dest_folder + file_name_suffix
    filename = os.path.join(dir_path, filename)

    # param_data = data_reading(file)
    X_1gtrain = train_set.get('X')
    y_train = train_set.get('Y')
    weights_train = train_set.get('benchmarks_weights')[0]

    X_1gtest = test_set.get('X')
    y_test = test_set.get('Y')
    weights_test = train_set.get('benchmarks_weights')[0]

    X_1gvalidation = X_1gtrain[0:2880, :]
    y_validation = y_train[0:2880]
    weights_validation = train_set.get('benchmarks_weights')[0]

    y_train = y_train[2880:]
    X_1gtrain = X_1gtrain[2880:, :]

    test_X_bench = test_set.get('benchmarks_X')
    test_X_bench = [benchmark[1:, :] for benchmark in test_X_bench]

    test_Y_bench = test_set.get('benchmarks_Y')
    test_weights_bench = test_set.get('benchmarks_weights')
    # print(test_X_bench[0])
    test_MSE_X = test_set.get('X')
    test_MSE_Y = test_set.get('Y')

    param=opt_params

    baseline_model = PredictPlusOptModel(opt_params=opt_params)

    baseline_model.init_params_lin_regression(X_1gtrain[:, 1:], y_train)

    start_time = time.time()
    #
    mypool = mp.Pool(processes=8)
    regret_baseline = baseline_model.get_regret(test_X_bench, test_Y_bench, test_weights_bench, pool=mypool)
    test_time = time.time() - start_time
    print('regret baseline', regret_baseline, 'time', test_time)
    mypool.close()

    ## Intopt HSD

    clf = intopt_energy(input_size=X_1gtrain.shape[1], param=param, hidden_size=1,
                        optimizer=optim.Adam, lr=0.7, num_layers=1, epochs=epoch_limit,
                        damping=1e-6, thr=0.1, validation_relax=False, store_validation=True)
    test_rslt=clf.fit_knapsack(X=X_1gtrain, y=y_train, weights=weights_train, X_validation=X_1gvalidation,y_validation=y_validation, weights_validation=weights_validation,
                               X_test=X_1gtest,y_test=y_test, weights_test=weights_test)
    rslt = test_rslt


    filename_0layer = os.path.join(dest_folder,"0l"+file_name_suffix)
    with open(filename_0layer, 'a') as f:
        rslt.to_csv(f, index=False, header=f.tell() == 0)

    # layer-1
    # # twostage
    # clf = twostage_energy(input_size=X_1gtrain.shape[1], param=param, hidden_size=100,
    #                       optimizer=optim.Adam, lr=0.01, num_layers=2, epochs=15, validation_relax=False)
    # clf.fit(X_1gtrain, y_train)
    # test_rslt = clf.validation_result(X_1gtest, y_test)
    #
    # two_stage_rslt = {'model': 'Two-stage', 'MSE-loss': test_rslt[1], 'Regret': test_rslt[0]}
    #
    # # SPO
    # clf = SPO_energy(input_size=X_1gtrain.shape[1], param=param, hidden_size=100,
    #                  optimizer=optim.Adam, lr=0.1, num_layers=2, epochs=5, validation_relax=False)
    # clf.fit(X_1gtrain, y_train)
    # test_rslt = clf.validation_result(X_1gtest, y_test)
    # spo_rslt = {'model': 'SPO', 'MSE-loss': test_rslt[1], 'Regret': test_rslt[0]}

    # Intopt HSD
    clf = intopt_energy(input_size=X_1gtrain.shape[1], param=param, hidden_size=100,
                        optimizer=optim.Adam, lr=0.1, num_layers=2, epochs=epoch_limit,
                        damping=0.00001, thr=0.1, validation_relax=False, store_validation=True)
    test_rslt=clf.fit_knapsack(X=X_1gtrain, y=y_train, weights=weights_train, X_validation=X_1gvalidation,y_validation=y_validation, weights_validation=weights_validation,
                               X_test=X_1gtest,y_test=y_test, weights_test=weights_test)
    rslt = test_rslt

    filename_1layer = os.path.join(dest_folder,"1l"+file_name_suffix)
    with open(filename_1layer, 'a') as f:
        rslt.to_csv(f, index=False, header=f.tell() == 0)
