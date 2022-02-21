'''

Takes multiple input arguments:
  (1) max_depth_set_str: sequence of training depths tuned using cross validation, e.g. "2-4-5"
  (2) min_samples_leaf_set_str: sequence of "min. (weighted) observations per leaf" tuned using cross validation, e.g. "20-50-100"
  (3) n_estimators_set_str: sequence of number of trees in forest tuned using cross validation, e.g. "20-50-100"
  (4) max_features_set_str: sequence of number of features used in feature bagging tuned using cross validation, e.g. "2-3-4"
  (5) algtype: set equal to "MSE" (CART forest) or "SPO" (SPOT forest)
  (6) number of workers to use in parallel processing (i.e., fitting individual trees in the forest in parallel)
  (7) decision_problem_seed: seed controlling generation of constraints in article recommendation problem (-1 = no constraints)
  (8) train_size: number of random obs. to extract from the training data.
  Only useful in limiting the size of the training data (-1 = use full training data)
  (9) quant_discret: continuous variable split points in the trees are chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc..

'''
import csv
import os
import time

import numpy as np
import pickle
# from gurobipy import *
from SPOTree_knapsack.decision_problem_solver import *
from SPOTree_knapsack.SPOForest import SPOForest

from SPOTree_knapsack.decision_problem_solver import find_opt_decision
from dnl.EnergyDataUtil import get_energy_data
from dnl.Utils import get_train_test_split_spotree

import sys

# SEEDS FOR RANDOM NUMBER GENERATORS
# seed for rngs in random forest
forest_seed = 0
# seed controlling random subset of training data used (if full training data is not being used)
select_train_seed = 0
#############################################################################
#############################################################################
#############################################################################

def forest_traintest(train_x, train_cost, test_x, test_cost, n_estimators, max_depth,
                     min_samples_leaf, max_features, run_in_parallel, num_workers, algtype, quant_discret, item_weights, capacity):
    if algtype == "MSE":
        SPO_weight_param = 0.0
    elif algtype == "SPO":
        SPO_weight_param = 1.0
    regr = SPOForest(n_estimators=n_estimators, run_in_parallel=run_in_parallel, num_workers=num_workers,
                     max_depth=max_depth, min_weights_per_node=min_samples_leaf, quant_discret=quant_discret,
                     debias_splits=False,
                     max_features=max_features,
                     SPO_weight_param=SPO_weight_param, SPO_full_error=True)
    train_cost = np.array(train_cost)
    train_x = np.array(train_x)
    # print('train x', train_x.shape)
    regr.fit(train_x, train_cost, verbose_forest=True, verbose=False, feats_continuous=True,
             seed=forest_seed, item_weights=item_weights, capacity=capacity)
    print('x', test_x.shape)
    pred_decision_mean = regr.est_decision(test_x, method="mean", Y=test_cost)
    print('pred dec', len(pred_decision_mean))
    # pred_decision_mode = regr.est_decision(test_x, method="mode")
    alg_costs_mean = [np.sum(test_cost[i] * pred_decision_mean[i]) for i in range(0, len(pred_decision_mean))]
    # alg_costs_mode = [np.sum(test_cost[i] * pred_decision_mode[i]) for i in range(0, len(pred_decision_mode))]
    return regr, np.median(alg_costs_mean)


def forest_tuneparams(train_x, train_cost, valid_x, valid_cost, n_estimators_set,
                      max_depth_set, min_samples_leaf_set, max_features_set, run_in_parallel, num_workers, algtype, quant_discret,
                      item_weights, capacity):
    best_err_mean = np.float("inf")
    best_err_mode = np.float("inf")
    run_time =0
    for n_estimators in n_estimators_set:
        for max_depth in max_depth_set:
            for min_samples_leaf in min_samples_leaf_set:
                for max_features in max_features_set:
                    start = time.time()
                    regr, err_mean = forest_traintest(train_x, train_cost, valid_x, valid_cost,
                                                      n_estimators, max_depth,
                                                      min_samples_leaf, max_features, run_in_parallel,
                                                      num_workers, algtype,quant_discret, item_weights, capacity)
                    end = time.time()
                    run_time = end-start
                    print('best_err_mean', best_err_mean, err_mean)
                    if err_mean <= best_err_mean:
                        best_run_time=run_time
                        best_regr_mean, best_err_mean, best_n_estimators_mean, best_max_depth_mean, best_min_samples_leaf_mean, best_max_features_mean = regr, err_mean, n_estimators, max_depth, min_samples_leaf, max_features
                    # if err_mode <= best_err_mode:
                    #     best_regr_mode, best_err_mode, best_n_estimators_mode, best_max_depth_mode, best_min_samples_leaf_mode, best_max_features_mode = regr, err_mode, n_estimators, max_depth, min_samples_leaf, max_features

    print("Best n_estimators (mean method): " + str(best_n_estimators_mean))
    print("Best max_depth (mean method): " + str(best_max_depth_mean))
    print("Best min_samples_leaf (mean method): " + str(best_min_samples_leaf_mean))
    print("Best max_features (mean method): " + str(best_max_features_mean))

    # print("Best n_estimators (mode method): " + str(best_n_estimators_mode))
    # print("Best max_depth (mode method): " + str(best_max_depth_mode))
    # print("Best min_samples_leaf (mode method): " + str(best_min_samples_leaf_mode))
    # print("Best max_features (mode method): " + str(best_max_features_mode))

    return best_regr_mean, best_err_mean, best_n_estimators_mean, best_max_depth_mean, best_min_samples_leaf_mean, best_max_features_mean, best_run_time
    # best_regr_mode, best_err_mode, best_n_estimators_mode, best_max_depth_mode, best_min_samples_leaf_mode, best_max_features_mode
########################################
# python SPOForest_knapsack.py "2-4-5" "20-50-100" "20-50-100" "2-3-4" "SPO" 1 -1 -1 0.05
def SPOTree_knapsack_wrapper(max_depth_set_str="5", min_samples_leaf_set_str="20",
                             n_estimators_set_str="50", max_features_set_str="5", algtype="SPO", core_number=4,
                             decision_problem_seed=-1, train_size=-1, quant_discret=0.05, capacity = 24, unit_weight=False, kfold = 4, file_folder = "Tests/spotree/"):

    ########################################
    # training parameters
    max_depth_set_str = max_depth_set_str
    max_depth_set = [int(k) for k in max_depth_set_str.split('-')]  # [None]
    min_samples_leaf_set_str = min_samples_leaf_set_str
    min_samples_leaf_set = [int(k) for k in min_samples_leaf_set_str.split('-')]  # [5]
    n_estimators_set_str = n_estimators_set_str
    n_estimators_set = [int(k) for k in n_estimators_set_str.split('-')]  # [100,500]
    max_features_set_str = max_features_set_str
    max_features_set = [int(k) for k in max_features_set_str.split('-')]  # [3]
    algtype = algtype  # either "MSE" or "SPO"
    # number of workers
    if core_number == "1":
        run_in_parallel = False
        num_workers = None
    else:
        run_in_parallel = True
        num_workers = int(core_number)
    # ineligible_seeds = [27,28,29,32,39] #considered range = 10-40 inclusive
    decision_problem_seed = int(decision_problem_seed)  # if -1, use no constraints in decision problem
    train_size = int(train_size)  # if you want to limit the size of the training data (-1 = no limit)
    quant_discret = float(quant_discret)
    capacity =  capacity
    ########################################
    # output filename
    #   fname_out_mode = algtype + "Forest_news_costs_depthSet" + max_depth_set_str + "_minObsSet" + min_samples_leaf_set_str + "_nEstSet" + n_estimators_set_str + "_mFeatSet" + max_features_set_str + "_aMethod" + "mode" + "_probSeed" + str(
    #     decision_problem_seed) + "_nTrain" + str(train_size) + "_qd" + str(quant_discret) + ".pkl";
    #############################################################################
    #############################################################################
    #############################################################################

    # generate decision problem
    num_constr = 5  # number of Aw <= b constraints
    num_dec = 6  # number of decisions
    # ineligible_seeds = [27,28,29,32,39] #considered range = 10-40 inclusive
    if decision_problem_seed == -1:
        # no budget constraint case
        A_constr = np.zeros((num_constr, num_dec))
    else:
        np.random.seed(decision_problem_seed)
        A_constr = np.random.exponential(scale=1.0, size=(num_constr, num_dec))


    ##############################################

    thresh = "50"
    valid_size = "50.0%"

    dataset = get_energy_data('../data/energy_data.txt', generate_weight=True, unit_weight=unit_weight, kfold=kfold, is_spo_tree=True)

    train_set, test_set = get_train_test_split_spotree(dataset, random_seed=42, is_shuffle=True)


    benchmarks_train_X = train_set.get('benchmarks_X')
    benchmarks_train_Y = train_set.get('benchmarks_Y')
    benchmarks_weights_train = train_set.get('benchmarks_weights')

    benchmarks_val_X = benchmarks_train_X[0:60]
    benchmarks_val_Y = benchmarks_train_Y[0:60]
    benchmarks_weights_val = benchmarks_weights_train[0:60]

    benchmarks_train_X = benchmarks_train_X[60:]
    benchmarks_train_Y = benchmarks_train_Y[60:]
    benchmarks_weights_train = benchmarks_weights_train[60:]

    test_X = test_set.get('benchmarks_X')
    test_Y = test_set.get('benchmarks_Y')
    test_weights = test_set.get('benchmarks_weights')

    train_x = benchmarks_train_X
    valid_x = benchmarks_val_X
    test_x = test_X

    # make negative to turn into minimization problem
    train_cost = benchmarks_train_Y
    valid_cost = benchmarks_val_Y
    test_cost = test_Y

    train_weights = benchmarks_weights_train
    valid_weights = benchmarks_weights_val
    test_weights = test_weights

    ##############################################
    # limit size of training data if specified
    if train_size != -1 and train_size <= train_x.shape[0] and train_size <= valid_x.shape[0]:
        np.random.seed(select_train_seed)
        sel_inds = np.random.choice(range(train_x.shape[0]), size=train_size, replace=False)
        train_x = train_x[sel_inds]
        train_cost = train_cost[sel_inds]
        train_weights = train_weights[sel_inds]
        sel_inds = np.random.choice(range(valid_x.shape[0]), size=train_size, replace=False)
        valid_x = valid_x[sel_inds]
        valid_cost = valid_cost[sel_inds]
        valid_weights = valid_weights[sel_inds]


    # FIT FOREST
    regr_mean, best_err_mean, best_n_estimators_mean, best_max_depth_mean, best_min_samples_leaf_mean, best_max_features_mean, best_run_time = forest_tuneparams(train_x, train_cost, valid_x, valid_cost, n_estimators_set,
                                                                           max_depth_set, min_samples_leaf_set,
                                                                           max_features_set, run_in_parallel, num_workers,
                                                                           algtype, quant_discret, test_weights[0], capacity)





    print(test_x)
    # FIND TEST SET COST
    pred_decision_mean = regr_mean.est_decision(test_x, item_weights=test_weights[0], method="mean", Y=test_cost)
    # pred_decision_mode = regr_mode.est_decision(test_x, method="mode")
    true__preds = find_opt_decision(values_arr=test_Y,item_weights=test_weights[0],capacity=capacity)['weights']
    true_costs_mean = np.array([np.sum(test_cost[i] * true__preds[i]) for i in range(0, len(true__preds))])
    costs_mean = np.array([np.sum(test_cost[i] * pred_decision_mean[i]) for i in range(0, len(pred_decision_mean))])
    # costs_mode = [np.sum(test_cost[i] * pred_decision_mode[i]) for i in range(0, len(pred_decision_mode))]
    print(len(costs_mean))
    print("Average test set cost (mean method) (max is better): " + str(np.median(true_costs_mean-costs_mean)))
    # print "Average test set cost (mode method) (max is better): " + str(-1.0 * np.mean(costs_mode))
    # print "Average test set weighted cost (mode method) (max is better): " + str(
    #     -1.0 * np.dot(costs_mode, test_weights) / np.sum(test_weights))

    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if unit_weight:
        unit_string = 'u'
    else:
        unit_string = ""
    fname_out_mean = "spotree_"+unit_string+"knap_c"+str(capacity)+"k"+str(kfold)+".csv";

    fname = file_folder + fname_out_mean
    fname = os.path.join(dir_path, fname)
    regret = np.median(true_costs_mean-costs_mean)
    save_output_spotree_knap(fname, best_err_mean, best_n_estimators_mean, best_max_depth_mean, best_min_samples_leaf_mean, best_max_features_mean, best_run_time, regret)


    # with open(fname_out_mode, 'wb') as output:
    #     pickle.dump(costs_mode, output, pickle.HIGHEST_PROTOCOL)

    # Getting back the objects:
    # with open(fname_out, 'rb') as input:
    #  costs_deg = pickle.load(input)

def save_output_spotree_knap(fname, best_err_mean, best_n_estimators_mean, best_max_depth_mean, best_min_samples_leaf_mean, best_max_features_mean, run_time, regret):
    first_line = ['n estimator', 'Max Depth', 'min samples', 'max features', 'run time', 'regret']
    second_line = [best_n_estimators_mean, best_max_depth_mean, best_min_samples_leaf_mean,
                   best_max_features_mean, run_time, regret]
    print = []
    print.append(first_line)
    print.append(second_line)
    with open(fname, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')

        for  value in print:
            csvwriter.writerow(value)
if __name__ == '__main__':
    SPOTree_knapsack_wrapper()