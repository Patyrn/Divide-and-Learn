
from qptl.qptl_model import *
import os
import sys
from dnl.PredictPlusOptimize import PredictPlusOptModel
random_seed = 42
sys.path.insert(0, '../../EnergyCost/')
sys.path.insert(0, "../..")
from SPO.ICON import *
import time, datetime
import logging


def qptl_ICON_wrapper(train_set, test_set, n_iter=10, instance_number=1, opt_params=None, file_name_suffix='empty',
              dest_folder="Tests/icon/Easy/kfolds/qptl/", time_limit=12000, epoch_limit=1):
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='../QPTL_Load2.log', level=logging.INFO, format=formatter)
    logging.info('Started\n')

    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file = "data/icon_instances/easy/instance" + str(instance_number) + ".txt"
    file = os.path.join(dir_path, file)

    filename = dest_folder + str(instance_number) + file_name_suffix
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
    # mypool = mp.Pool(processes=8)
    # regret_baseline = baseline_model.get_regret(test_X_bench, test_Y_bench, test_weights_bench, pool=mypool)
    # test_time = time.time() - start_time
    # print('regret baseline', regret_baseline, 'time', test_time)
    # mypool.close()







    print('insatance', instance_number)
    H_combinations = [{'lr': 1e-2}, {'optimizer': optim.Adam, 'lr': 1e-2}]
    for i in range(n_iter):
        for h in H_combinations:
            print("hyperparams : %s  Time:%s \n" % (str(h), datetime.datetime.now()))
            start = time.time()
            clf = qptl_ICON(epochs=epoch_limit, param=param_data, verbose=True, validation=True, validation_relax=True, **h)
            pdf = clf.fit(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test)
            end = time.time()
            pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
            pdf['total_time'] = end - start
            pdf['validation_relax'] = True
            with open(filename, 'a') as f:
                pdf.to_csv(f, mode='a', header=f.tell() == 0, index=False)

