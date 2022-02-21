import os
import sys

from dnl.PredictPlusOptimize import PredictPlusOptModel

random_seed = 42
sys.path.insert(0, '../../EnergyCost/')
sys.path.insert(0, "../..")
from SPO.torch_SPO_updated import *
from SPO.ICON import *
import time, datetime
import logging


def SPO_load2(train_set, test_set, n_iter=10, instance_number=1, opt_params=None, file_name_suffix='empty',
              dest_folder="Tests/icon/Easy/kfolds/spo/", time_limit=12000, epoch_limit=1):
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='SPO/SPO_Load1.log', level=logging.INFO, format=formatter)
    logging.info('Started\n')

    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file = "data/icon_instances/easy/instance" + str(instance_number) + ".txt"
    file = os.path.join(dir_path, file)

    filename = dest_folder + file_name_suffix
    filename = os.path.join(dir_path, filename)

    param_data = data_reading(file)
    X_1gtrain = train_set.get('X')
    y_train = train_set.get('Y')
    X_1gtest = test_set.get('X')
    y_test = test_set.get('Y')

    X_1gvalidation = X_1gtrain[0:2880, :]
    y_validation = y_train[0:2880]

    y_train = y_train[2880:]
    X_1gtrain = X_1gtrain[2880:, :]

    test_X_bench = test_set.get('benchmarks_X')
    test_X_bench = [benchmark[1:, :] for benchmark in test_X_bench]

    test_Y_bench = test_set.get('benchmarks_Y')
    test_weights_bench = test_set.get('benchmarks_weights')
    # print(test_X_bench[0])
    test_MSE_X = test_set.get('X')
    test_MSE_Y = test_set.get('Y')

    baseline_model = PredictPlusOptModel(opt_params=opt_params)

    baseline_model.init_params_lin_regression(X_1gtrain[:, 1:], y_train)

    start_time = time.time()
    #
    mypool = mp.Pool(processes=8)
    regret_baseline = baseline_model.get_regret(test_X_bench, test_Y_bench, test_weights_bench, pool=mypool)
    test_time = time.time() - start_time
    print('regret baseline', regret_baseline, 'time', test_time)
    mypool.close()

    H_list = [{'lr': 1e-4, 'momentum': 0.01}, {'lr': 1e-4}]

    warmstart_hyperparams = [{'reset': True, 'presolve': False, 'warmstart': False},
                             {'reset': True, 'presolve': True, 'warmstart': False},
                             {'reset': False, 'presolve': True, 'warmstart': False},
                             {'reset': False, 'presolve': True, 'warmstart': True}]



    print('insatance', instance_number)
    for w in warmstart_hyperparams:
        for h in H_list:
            for i in range(n_iter):
                print("N : %d Time:%s %s %s\n" % (i, datetime.datetime.now(), h, w))
                clf = SGD_SPO_generic(solver=Gurobi_ICON, accuracy_measure=False, relax=True,
                                      validation_relax=True, verbose=True, param=param_data, maximize=True,
                                      epochs=epoch_limit, timelimit=time_limit, **h, **w)
                start = time.time()
                pdf = clf.fit(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test)
                end = time.time()
                pdf['training_relaxation'] = True
                pdf['validation_relaxation'] = True
                pdf['reset'] = w['reset']
                pdf['presolve'] = w['presolve']
                pdf['warmstart'] = w['warmstart']
                pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
                with open(filename, 'a') as f:
                    pdf.to_csv(f, mode='a', header=f.tell() == 0, index=False)
