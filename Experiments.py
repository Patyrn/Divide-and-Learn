import random

import numpy as np
import multiprocessing as mp

from IntOpt import intopt_icon
from IntOpt import exp_run
from IntOpt.exp_run import exp_run_wow
from IntOpt.intopt_icon import intopt_icon_run
from IntOpt.intopt_weighted_knapsack import weighted_knapsack_intopt
from SPOTree_scheduling.SPOForest_scheduling import SPOTree_scheduling
from dnl import Sampling_Methods
from dnl.EnergyDataUtil import get_energy_data
from SPO.ICON_Load2SPO import SPO_load2
from dnl.IconEasySolver import get_icon_instance_params
from dnl.KnapsackSolver import get_opt_params_knapsack
from dp.PredictOptDP import PredictOptDP
from dnl.PredictPlusOptimize import PredictPlusOptModel
from dnl.Utils import get_train_test_split, get_train_test_split_SPO, save_results_list
from qptl.Weighted_knapsack_qptl import weighted_knapsack_qptl
from SPO.Weighted_knapsack_spo import weighted_knapsack_SPO
from qptl.qptl_ICON import qptl_ICON_wrapper
from SPOTree_knapsack.SPOForest_knapsack import SPOTree_knapsack_wrapper


def train_and_test(file_name_prefix='noprefix', file_folder='', max_step_size_magnitude=0, min_step_size_magnitude=-1,
                   step_size_divider=10, opt_params=None,
                   generate_weight=True, unit_weight=True, is_shuffle=False, print_test=True,
                   test_boolean=None, core_number=7, time_limit=12000, epoch_limit=3, mini_batch_size=32, verbose=False,
                   kfold=0, learning_rate=0.3, dataset=None, noise_level=0, beta=0):
    random.seed(42)
    NUMBER_OF_RANDOM_TESTS = 1
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    # random_seeds = [42 for p in range(NUMBER_OF_RANDOM_TESTS)]
    global exhaustive_running_time_max, divide_conquer_time_max, divide_conquer_greedy_time, divide_greedy_profit, divide_profit_max, divide_profit, exhaustive_profit_max, exhaustive_profit, divide_conquer_time, exhaustive_running_time
    if test_boolean is None:
        test_boolean = [0, 0, 0, 0, 1]
    NUMBER_OF_MODELS = 5
    exhaustive_model = 0
    exhaustive_max_model = 1
    divide_and_conquer_model = 2
    divide_and_conquer_max_model = 3
    divide_and_conquer_greedy_model = 4
    baseline_regression = 5

    training_obj_values_per_epoch = [[[] for n in range(NUMBER_OF_MODELS)] for j in range(NUMBER_OF_RANDOM_TESTS)]
    training_obj_values = np.zeros((NUMBER_OF_MODELS + 1, NUMBER_OF_RANDOM_TESTS))
    epochs = np.zeros((NUMBER_OF_MODELS, NUMBER_OF_RANDOM_TESTS))
    regrets = np.zeros((NUMBER_OF_MODELS + 1, NUMBER_OF_RANDOM_TESTS))
    run_times = np.zeros((NUMBER_OF_MODELS, NUMBER_OF_RANDOM_TESTS))
    test_MSES = np.zeros((NUMBER_OF_MODELS + 1, NUMBER_OF_RANDOM_TESTS))
    training_MSES = np.zeros((NUMBER_OF_MODELS, NUMBER_OF_RANDOM_TESTS))  # might not use

    model_method_names = ['Exhaustive',
                          'Exhaustive Select Max',
                          'Divide and Conquer',
                          'Divide and Conquer Select Max',
                          'Divide and Conquer Select Greedy'
                          ]
    if dataset is None:
        dataset = get_energy_data('energy_data.txt', generate_weight=generate_weight, unit_weight=unit_weight,
                                  kfold=kfold, noise_level=noise_level)

    # combine weights with X first
    # may need to split weights
    for random_test_index, random_seed in zip(range(NUMBER_OF_RANDOM_TESTS), random_seeds):
        train_set, test_set = get_train_test_split(dataset, random_seed=random_seed, is_shuffle=is_shuffle)

        X_train = train_set.get('X')
        Y_train = train_set.get('Y')

        X_val = X_train[0:2880, :]
        Y_val = Y_train[0:2880]

        X_train = X_train[2880:]
        Y_train = Y_train[2880:, :]

        benchmarks_train_X = train_set.get('benchmarks_X')
        benchmarks_train_Y = train_set.get('benchmarks_Y')
        benchmarks_weights_train = train_set.get('benchmarks_weights')

        benchmarks_val_X = benchmarks_train_X[0:60]
        benchmarks_val_Y = benchmarks_train_Y[0:60]
        benchmarks_weights_val = benchmarks_weights_train[0:60]

        benchmarks_train_X = benchmarks_train_X[60:]
        benchmarks_train_Y = benchmarks_train_Y[60:]
        benchmarks_weights_train = benchmarks_weights_train[60:]

        # benchmark_number = 1
        train_X = benchmarks_train_X
        train_Y = benchmarks_train_Y
        train_weights = benchmarks_weights_train

        val_X = benchmarks_val_X
        val_Y = benchmarks_val_Y
        val_weights = benchmarks_weights_val

        test_X = test_set.get('benchmarks_X')
        test_Y = test_set.get('benchmarks_Y')
        test_weights = test_set.get('benchmarks_weights')
        #
        test_MSE_X = test_set.get('X')
        test_MSE_Y = test_set.get('Y')
        mypool = mp.Pool(processes=8)
        baseline_model = PredictPlusOptModel(opt_params=opt_params)
        baseline_model.init_params_lin_regression(X=X_train, Y=Y_train)
        baseline_model.get_regret(test_X, test_Y, test_weights, pool=mypool)
        mypool.close()

        print('baseline test regret:', baseline_model.test_regret)

        models = []

        models.append(PredictPlusOptModel(sampling_method=Sampling_Methods.EXHAUSTIVE,
                                          min_step_size_magnitude=min_step_size_magnitude,
                                          max_step_size_magnitude=max_step_size_magnitude,
                                          step_size_divider=step_size_divider, opt_params=opt_params,
                                          learning_rate=learning_rate, beta=beta, mini_batch_size=mini_batch_size,
                                          run_time_limit=time_limit, epoch_limit=epoch_limit, verbose=verbose))

        models.append(PredictPlusOptModel(sampling_method=Sampling_Methods.EXHAUSTIVE_MAX,
                                          min_step_size_magnitude=min_step_size_magnitude,
                                          max_step_size_magnitude=max_step_size_magnitude,
                                          step_size_divider=step_size_divider, opt_params=opt_params,
                                          learning_rate=learning_rate, beta=beta, mini_batch_size=mini_batch_size,
                                          run_time_limit=time_limit, epoch_limit=epoch_limit, verbose=verbose))

        models.append(PredictPlusOptModel(sampling_method=Sampling_Methods.DIVIDE_AND_CONQUER,
                                          min_step_size_magnitude=min_step_size_magnitude,
                                          max_step_size_magnitude=max_step_size_magnitude,
                                          step_size_divider=step_size_divider, opt_params=opt_params,
                                          learning_rate=learning_rate, beta=beta, mini_batch_size=mini_batch_size,
                                          run_time_limit=time_limit, epoch_limit=epoch_limit, verbose=verbose))

        models.append(PredictPlusOptModel(sampling_method=Sampling_Methods.DIVIDE_AND_CONQUER_MAX,
                                          min_step_size_magnitude=min_step_size_magnitude,
                                          max_step_size_magnitude=max_step_size_magnitude,
                                          step_size_divider=step_size_divider, opt_params=opt_params,
                                          learning_rate=learning_rate, beta=beta, mini_batch_size=mini_batch_size,
                                          run_time_limit=time_limit, epoch_limit=epoch_limit, verbose=verbose))

        models.append(PredictPlusOptModel(
            sampling_method=Sampling_Methods.DIVIDE_AND_CONQUER_GREEDY,
            min_step_size_magnitude=min_step_size_magnitude,
            max_step_size_magnitude=max_step_size_magnitude,
            step_size_divider=step_size_divider, opt_params=opt_params,
            learning_rate=learning_rate, beta=beta, mini_batch_size=mini_batch_size,
            run_time_limit=time_limit, epoch_limit=epoch_limit, verbose=verbose))

        # initialize models

        for model in models:
            model.init_params_lin_regression(X=X_train, Y=Y_train)
        # #
        # run coordinate descent
        for model, i in zip(models, range(NUMBER_OF_MODELS)):
            if test_boolean[i] == True:
                print("Starting", model_method_names[i], 'Model')

                model.coordinate_descent(train_X=train_X, train_Y=train_Y,
                                         train_weights=train_weights, val_X=val_X, val_Y=val_Y, val_weights=val_weights,
                                         test_X=test_X, test_Y=test_Y,
                                         test_weights=test_weights, print_test=print_test, core_number=core_number)
                print(model_method_names[i], "Running Time:", str(model.run_time[-1]) + "s\n")
                print(model_method_names[i], "Test Running Time:", str(model.test_run_time) + "s\n")
                file_name = file_name_prefix + model.get_file_name()
                model.get_MSE(test_MSE_X, test_MSE_Y)
                save_results_list(file_folder=file_folder, file_name=file_name, results=model.print())

        print("----RESULTS----")

        for model, i in zip(models, range(NUMBER_OF_MODELS)):

            if test_boolean[i] == True:
                print(model_method_names[i], 'Objective Value:', model.training_obj_value[-1], "Running Time:",
                      str(model.run_time[-1]) + "s\n")

                run_times[i, random_test_index] = model.run_time[-1]
                training_obj_values_per_epoch[random_test_index][i].extend(model.training_obj_value)
                training_obj_values[i, random_test_index] = model.training_obj_value[-1]
                epochs[i, random_test_index] = model.number_of_epochs
                print(model_method_names[i], 'Objective Value:', model.training_obj_value[-1], "Running Time:",
                      str(model.run_time[-1]) + "s\n")

                training_obj_values[baseline_regression, random_test_index] = model.training_obj_value[0]

        print("----END----")

        # Tests
        baseline_model.get_MSE(test_MSE_X, test_MSE_Y)

        regrets[baseline_regression, random_test_index] = baseline_model.test_regret
        test_MSES[baseline_regression, random_test_index] = baseline_model.test_MSE

        print('printing regret baseline = ' + str(baseline_model.test_regret) + ', printing MSE baseline = ' + str(
            baseline_model.test_MSE))

        for model, i in zip(models, range(NUMBER_OF_MODELS)):
            if test_boolean[i] == True:
                model.get_regret(test_X, test_Y, test_weights)
                model.get_MSE(test_MSE_X, test_MSE_Y)
                print(model_method_names[i], 'Regret:', model.test_regret, "MSE:",
                      str(model.test_MSE) + "s\n")
                regrets[i, random_test_index] = model.test_regret
                test_MSES[i, random_test_index] = model.test_MSE
    print('printing regret baseline = ' + str(
        np.mean(regrets[baseline_regression, :])) + ', printing MSE baseline = ' + str(
        np.mean(test_MSES[baseline_regression, :])))
    for model, i in zip(models, range(NUMBER_OF_MODELS)):
        if test_boolean[i] == True:
            print(model_method_names[i], 'Regret:', np.mean(regrets[i, :]), "MSE:",
                  str(np.mean(test_MSES[i, :])) + "s", "Running time", np.mean(run_times[i, :]))


def train_and_test_DP(opt_params=None,
                      generate_weight=True, unit_weight=True,
                      print_test=True, kfold=0, file_folder="", core_number=7, noise_level=0, dataset=None,
                      is_shuffle=False, file_name_prefix=None):
    if dataset is None:
        dataset = get_energy_data('energy_data.txt', generate_weight=generate_weight, unit_weight=unit_weight,
                                  kfold=kfold, noise_level=noise_level)

    train_set, test_set = get_train_test_split(dataset, random_seed=42, is_shuffle=is_shuffle)

    X_train = train_set.get('X')
    Y_train = train_set.get('Y')

    X_val = X_train[0:2880, :]
    Y_val = Y_train[0:2880]

    X_train = X_train[2880:]
    Y_train = Y_train[2880:, :]

    benchmarks_train_X = train_set.get('benchmarks_X')
    benchmarks_train_Y = train_set.get('benchmarks_Y')
    benchmarks_weights_train = train_set.get('benchmarks_weights')

    benchmarks_val_X = benchmarks_train_X[0:60]
    benchmarks_val_Y = benchmarks_train_Y[0:60]
    benchmarks_weights_val = benchmarks_weights_train[0:60]

    benchmarks_train_X = benchmarks_train_X[60:]
    benchmarks_train_Y = benchmarks_train_Y[60:]
    benchmarks_weights_train = benchmarks_weights_train[60:]

    # benchmark_number = 1
    train_X = benchmarks_train_X
    train_Y = benchmarks_train_Y
    train_weights = benchmarks_weights_train

    val_X = benchmarks_val_X
    val_Y = benchmarks_val_Y
    val_weights = benchmarks_weights_val

    test_X = test_set.get('benchmarks_X')
    test_Y = test_set.get('benchmarks_Y')
    test_weights = test_set.get('benchmarks_weights')
    #
    test_MSE_X = test_set.get('X')
    test_MSE_Y = test_set.get('Y')

    # Initialize parameters and get coefficients

    baseline_model = PredictPlusOptModel(opt_params=opt_params)

    # initialize models
    baseline_model.init_params_lin_regression(X_train, Y_train)

    print('baseline regret: {}'.format(baseline_model.get_regret(X=test_X, Y=test_Y, weights=test_weights)))

    #
    # divide_and_conquer_greedy_model = PredictPlusOptModel(
    #     sampling_method=Sampling_Methods.DIVIDE_AND_CONQUER_GREEDY,
    #     min_step_size_magnitude=-1,
    #     max_step_size_magnitude=0,
    #     step_size_divider=10, opt_params=opt_params,
    #     learning_rate=0.1, beta=0, mini_batch_size=32,
    #     run_time_limit=12000, epoch_limit=3, verbose=False)
    #
    # divide_and_conquer_greedy_model.init_params_lin_regression(X_train, Y_train)
    #
    # divide_greedy_profit = divide_and_conquer_greedy_model.coordinate_descent(train_X=train_X, train_Y=train_Y,
    #                                                                           train_weights=train_weights, val_X=val_X,
    #                                                                           val_Y=val_Y, val_weights=val_weights,
    #                                                                           test_X=test_X, test_Y=test_Y,
    #                                                                           test_weights=test_weights,
    #                                                                           print_test=print_test,
    #                                                                           core_number=7)

    dp_model = PredictOptDP(opt_params=opt_params, verbose=False)
    dp_model.init_params_lin_regression(X_train, Y_train)
    dp_model.coordinate_descent(train_X=train_X, train_Y=train_Y, train_weights=train_weights, test_X=test_X,
                                test_Y=test_Y, test_weights=test_weights, val_X=val_X, val_Y=val_Y,
                                val_weights=val_weights, print_test=print_test, core_number=core_number)
    file_name = file_name_prefix + dp_model.get_file_name()
    save_results_list(file_folder=file_folder, file_name=file_name, results=dp_model.print())


def test_Icon_unit(is_shuffle=True, max_step_size_magnitude=0, min_step_size_magnitude=-1, learning_rate=0.1,
                   loads=[12],
                   test_boolean=[0, 0, 0, 0, 1], core_number=7, kfolds=[0], n_iter=5, mini_batch_size=32, beta=0,
                   file_folder='Tests/icon/Easy/kfolds/val/spartan'):
    n_range = range(n_iter)
    for load in loads:
        for n in n_range:
            for kfold in kfolds:
                opt_params = get_icon_instance_params(load)
                file_name_prefix = 'iconmax-l' + str(load) + 'k' + str(kfold) + '-'
                train_and_test(file_name_prefix=file_name_prefix, file_folder=file_folder,
                               max_step_size_magnitude=max_step_size_magnitude,
                               min_step_size_magnitude=min_step_size_magnitude,
                               opt_params=opt_params,
                               generate_weight=True, unit_weight=True, core_number=core_number,
                               test_boolean=test_boolean, mini_batch_size=mini_batch_size, verbose=False,
                               is_shuffle=is_shuffle, kfold=kfold, learning_rate=learning_rate, beta=beta)


def test_SPO(instance_number=1, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
             dest_folder="Tests/icon/Easy/kfolds/spo/", time_limit=12000, epoch_limit=6):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold)
        opt_params = get_icon_instance_params(instance_number)

        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)
        # train_set, test_set = get_train_test_split(dataset, random_seed=random_seed, is_shuffle=is_shuffle)
        # print(train_set_SPO)
        # print(train_set)
        # baseline_model = PredictPlusOptModel(opt_params=opt_params)
        # baseline_model.init_params_lin_regression(train_set=train_set)

        # test_X = test_set.get('benchmarks_X')
        # test_Y = test_set.get('benchmarks_Y')
        # test_weights = test_set.get('benchmarks_weights')
        # print(len(test_X))
        # print(len(test_Y))
        # print(len(test_weights))
        # baseline_regret = baseline_model.get_regret(test_X, test_Y, test_weights)
        file_name_suffix = 'Load' + str(instance_number) + 'SPOmax_spartan_kfold' + str(kfold) + '.csv'
        # print("baseline_regret", baseline_regret)
        SPO_load2(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, opt_params=opt_params,
                  instance_number=instance_number, file_name_suffix=file_name_suffix, dest_folder=dest_folder,
                  time_limit=time_limit, epoch_limit=epoch_limit)

def test_intopt(instance_number=1, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=1,
             dest_folder="Tests/icon/intopt/", time_limit=12000, epoch_limit=1):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold)
        opt_params = get_icon_instance_params(instance_number)

        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)
        file_name_suffix = 'intoptl' + str(instance_number) + 'k' + str(kfold) + '.csv'
        intopt_icon_run(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, opt_params=opt_params,
                  instance_number=instance_number, file_name_suffix=file_name_suffix, dest_folder=dest_folder,
                  time_limit=time_limit, epoch_limit=epoch_limit)

def test_QPTL(instance_number=1, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
              dest_folder="Tests/icon/Easy/kfolds/qptl/", time_limit=12000, epoch_limit=6):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold)
        opt_params = get_icon_instance_params(instance_number)

        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)

        file_name_suffix = 'Load' + str(instance_number) + 'qptlmax_spartan_kfold' + str(kfold) + '.csv'

        qptl_ICON_wrapper(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, opt_params=opt_params,
                          instance_number=instance_number, file_name_suffix=file_name_suffix, dest_folder=dest_folder,
                          time_limit=time_limit, epoch_limit=epoch_limit)


def test_knapsack_SPO(capacities=[12], is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
                      dest_folder="Tests/icon/Easy/kfolds/spo/", noise_level=0):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False, kfold=kfold,
                                  noise_level=noise_level)
        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle
                                                               )

        file_name_suffix = 'knapsack_SPOk' + str(kfold)
        # print("baseline_regret", baseline_regret)
        weighted_knapsack_SPO(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, capacities=capacities,
                              file_name_suffix=file_name_suffix, dest_folder=dest_folder)


def test_knapsack_qptl(capacities=[12], is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
                       dest_folder="Tests/icon/Easy/kfolds/qptl/", noise_level=0):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False, kfold=kfold,
                                  noise_level=noise_level)
        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle
                                                               )

        file_name_suffix = 'knapsack_qptlk' + str(kfold)
        # print("baseline_regret", baseline_regret)
        weighted_knapsack_qptl(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, capacities=capacities,
                               file_name_suffix=file_name_suffix, dest_folder=dest_folder)


def test_knapsack_qptl_unit(capacities=[5], is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
                            dest_folder="Tests/icon/Easy/kfolds/qptl/", noise_level=0):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold,
                                  noise_level=noise_level)
        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed,
                                                               is_shuffle=is_shuffle
                                                               )

        file_name_suffix = 'knapsack_qptlk' + str(kfold)
        # print("baseline_regret", baseline_regret)
        weighted_knapsack_qptl(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO,
                               capacities=capacities,
                               file_name_suffix=file_name_suffix, dest_folder=dest_folder)

def test_knapsack_qptl_unit(capacities=[5], is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
                            dest_folder="Tests/icon/Easy/kfolds/qptl/", noise_level=0):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold,
                                  noise_level=noise_level)
        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed,
                                                               is_shuffle=is_shuffle
                                                               )

        file_name_suffix = 'knapsack_qptlk' + str(kfold)
        # print("baseline_regret", baseline_regret)
        weighted_knapsack_qptl(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO,
                               capacities=capacities,
                               file_name_suffix=file_name_suffix, dest_folder=dest_folder)



def test_knapsack_weighted(max_step_size_magnitude=0, min_step_size_magnitude=-1, capacities=[12], epoch_limit=3,
                           kfolds=[0],
                           test_boolean=[0, 0, 0, 0, 1], core_number=7, is_shuffle=False, learning_rate=0.1,
                           mini_batch_size=32, n_iter=5, noise_level=0, file_folder_prefix=None,beta=0.9):
    # dataset = np.load('Data.npz')
    dataset = None

    for capacity in capacities:
        if file_folder_prefix is None:
            file_folder = 'Tests/Knapsack/weighted/c' + str(capacity) + '/laptop/'
        else:
            file_folder = file_folder_prefix
        for n in range(n_iter):
            for kfold in kfolds:
                opt_params = get_opt_params_knapsack(capacity=capacity)
                file_name_prefix = 'N' + str(noise_level) + 'gurobi_knapsack-wsparc' + str(capacity) + "-k" + str(
                    kfold) + '-'
                train_and_test(dataset=dataset, kfold=kfold, file_name_prefix=file_name_prefix, file_folder=file_folder,
                               max_step_size_magnitude=max_step_size_magnitude,
                               min_step_size_magnitude=min_step_size_magnitude,
                               opt_params=opt_params, epoch_limit=epoch_limit, is_shuffle=is_shuffle,
                               generate_weight=True, unit_weight=False, core_number=core_number,
                               test_boolean=test_boolean,
                               learning_rate=learning_rate, mini_batch_size=mini_batch_size, noise_level=noise_level, beta=beta)

def test_intopt_knapsack(capacity=12, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=1,
             dest_folder="Tests/intopt/", time_limit=12000, epoch_limit=1):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False, kfold=kfold)
        opt_params = get_opt_params_knapsack(capacity=capacity)

        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)
        file_name_suffix = 'intoptc' + str(capacity) + 'k' + str(kfold) + '.csv'
        weighted_knapsack_intopt(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, opt_params=opt_params,
                  capacity=capacity, file_name_suffix=file_name_suffix, dest_folder=dest_folder,
                  time_limit=time_limit, epoch_limit=epoch_limit)

def test_intopt_unit_knapsack(capacity=12, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=1,
             dest_folder="Tests/intopt/", time_limit=12000, epoch_limit=1):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold)
        opt_params = get_opt_params_knapsack(capacity=capacity)

        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)
        file_name_suffix = 'intopt_unitc' + str(capacity) + 'k' + str(kfold) + '.csv'
        weighted_knapsack_intopt(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, opt_params=opt_params,
                  capacity=capacity, file_name_suffix=file_name_suffix, dest_folder=dest_folder,
                  time_limit=time_limit, epoch_limit=epoch_limit)


def test_knapsack_unit_DP(capacities=[12],
                          kfolds=[0],
                          core_number=7, is_shuffle=False,
                          n_iter=1, noise_level=0, file_folder_prefix=None):
    # dataset = np.load('Data.npz')
    dataset = None

    for capacity in capacities:
        if file_folder_prefix is None:
            file_folder = 'Tests/Knapsack/unit/c' + str(capacity) + '/dp/'
        else:
            file_folder = file_folder_prefix
        for n in range(n_iter):
            for kfold in kfolds:
                opt_params = get_opt_params_knapsack(capacity=capacity)
                file_name_prefix = 'knapsack-c' + str(capacity) + "-k" + str(kfold) + '-'
                train_and_test_DP(dataset=dataset, kfold=kfold, file_folder=file_folder,
                                  file_name_prefix=file_name_prefix,
                                  opt_params=opt_params,
                                  generate_weight=True, unit_weight=True, core_number=core_number,
                                  is_shuffle=is_shuffle, noise_level=noise_level)


def test_knapsack_weighted_DP(capacities=[12],
                              kfolds=[0],
                              core_number=7, is_shuffle=False,
                              n_iter=1, noise_level=0, file_folder_prefix=None):
    # dataset = np.load('Data.npz')
    dataset = None

    for capacity in capacities:
        if file_folder_prefix is None:
            file_folder = 'Tests/Knapsack/weighted/c' + str(capacity) + '/dp/'
        else:
            file_folder = file_folder_prefix
        for n in range(n_iter):
            for kfold in kfolds:
                opt_params = get_opt_params_knapsack(capacity=capacity)
                file_name_prefix = 'knapsack-wc' + str(capacity) + "-k" + str(kfold) + '-'
                train_and_test_DP(dataset=dataset, kfold=kfold, file_folder=file_folder,
                                  file_name_prefix=file_name_prefix,
                                  opt_params=opt_params,
                                  generate_weight=True, unit_weight=False, core_number=core_number,
                                  is_shuffle=is_shuffle, noise_level=noise_level)


def test_knapsack_unit(max_step_size_magnitude=0, min_step_size_magnitude=-1, capacities=[12],
                       epoch_limit=3, kfolds=[0],
                       test_boolean=[0, 0, 0, 0, 1], core_number=7, is_shuffle=False,
                       learning_rate=0.1, mini_batch_size=32, n_iter=5):
    # dataset = np.load('Data.npz')
    dataset = None

    for capacity in capacities:
        file_folder = 'Tests/Knapsack/unit/c' + str(capacity) + '/laptop/'
        for n in range(n_iter):
            for kfold in kfolds:
                opt_params = get_opt_params_knapsack(capacity=capacity)
                file_name_prefix = 'gurobi_knapsack-c' + str(capacity) + "-k" + str(kfold) + '-'
                train_and_test(dataset=dataset, kfold=kfold, file_name_prefix=file_name_prefix,
                               file_folder=file_folder,
                               max_step_size_magnitude=max_step_size_magnitude,
                               min_step_size_magnitude=min_step_size_magnitude,
                               opt_params=opt_params, epoch_limit=epoch_limit, is_shuffle=is_shuffle,
                               generate_weight=True, unit_weight=True, core_number=core_number,
                               test_boolean=test_boolean,
                               learning_rate=learning_rate, mini_batch_size=mini_batch_size)


def test_knapsack_SPO_unit(capacities=[12], is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
                           dest_folder="Tests/icon/Easy/kfolds/spo/"):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    for kfold in kfolds:
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold)

        train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)

        file_name_suffix = 'knapsack_SPOk' + str(kfold)
        # print("baseline_regret", baseline_regret)
        weighted_knapsack_SPO(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, capacities=capacities,
                              file_name_suffix=file_name_suffix, dest_folder=dest_folder)


if __name__ == '__main__':
    print('hello')