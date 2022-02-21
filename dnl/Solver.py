# from IconEasySolver import compute_profit_scheduling_easy, compute_optimal_average_value_icon_easy, \
#     compute_sampled_alpha_profit_icon_easy

import numpy as np

from dnl.IconEasySolver import compute_optimal_average_value_icon_energy, compute_icon_energy_cost, \
    compute_icon_energy_single_benchmark
from dnl.KnapsackSolver import compute_optimal_average_value_knapsack, \
    compute_profit_knapsack, compute_profit_knapsack_single_benchmark
from dnl.Params import KNAPSACK, ICON_SCHEDULING_EASY
from dnl.PredictPlustOptimizeUtils import compute_C_k, compute_F_k


def get_optimization_objective_for_samples(benchmark_X, benchmark_Y, benchmark_weights, opt_params, model_params,
                                           sample_space,
                                           k, mpPool=None):
    """
    Computes regrets of a single benchmark given a range of alpha samples
    :param benchmark_X:
    :param benchmark_Y:
    :param benchmark_weights:
    :param capacities:
    :param alphas: vector of all alpha
    :param const: regression constant
    :param sample_space: range of alphas
    :param k: indicator of the current alpha
    :return:sampled_profits(nparray), sampled_predicted_profits(nparray)
    """
    alphas = model_params.get('alphas')
    const = model_params.get('const')

    sampled_profits = []
    sampled_predicted_profits = []
    C_k = compute_C_k(benchmark_X, alphas, const, k)
    # for tmp_alpha in sample_space:

    # Compute predicted cost F_k for each alpha_k in searching space for the current k
    # serial

    for tmp_alpha in sample_space:
        profit, predicted_profit = compute_objective_value_single_benchmarks(tmp_alpha, benchmark_X,
                                                                             benchmark_Y,
                                                                             benchmark_weights, C_k,

                                                                             opt_params, k)
        sampled_profits.append(profit)
        sampled_predicted_profits.append(predicted_profit)

    # parallel

    # mypool = mp.Pool(processes=min(8, mp.cpu_count()))
    # mypool = mpPool
    #
    # map_func = partial(compute_objective_value_single_benchmarks, benchmark_X=benchmark_X, Y=benchmark_Y,
    #                    weights=benchmark_weights, C_k=C_k, opt_params=opt_params, k=k)
    # results = mypool.map(map_func, sample_space)
    #
    # for profit, predicted_profit in results:
    #     sampled_profits.append(profit)
    #     sampled_predicted_profits.append(predicted_profit)

    sampled_profits = np.array(sampled_profits)
    sampled_predicted_profits = np.array(sampled_predicted_profits)
    return sampled_profits, sampled_predicted_profits


def compute_objective_value_single_benchmarks(tmp_alpha, benchmark_X, Y, weights,
                                              C_k, opt_params, k):
    # Compute predicted cost F_k for each alpha_k in searching space for the current k
    F_k = compute_F_k(benchmark_X, tmp_alpha, C_k, k)

    sampled_profits = -1
    sampled_predicted_profits = -1
    solver = opt_params.get('solver')
    if solver == KNAPSACK:
        capacity = opt_params.get('capacity')
        sampled_profits, sampled_predicted_profits, __ = compute_profit_knapsack_single_benchmark(F_k, Y, weights,
                                                                                                  capacity)
    elif solver == ICON_SCHEDULING_EASY:
        sampled_profits, sampled_predicted_profits, __ = compute_icon_energy_single_benchmark(F_k, Y, opt_params)

    return sampled_profits, sampled_predicted_profits


#
# def get_optimization_objective_for_samples(benchmark_X, benchmark_Y, benchmark_weights, model_params, opt_params,
#                                            sample_space, k, solver='knapsack'):
#     sampled_profits = -1
#     sampled_predicted_profits = -1
#     solver = opt_params.get('solver')
#     if solver == KNAPSACK:
#         sampled_profits, sampled_predicted_profits, __ = compute_sampled_alpha_profit_knapsack(benchmark_X, benchmark_Y,
#                                                                                                benchmark_weights,
#                                                                                                opt_params,
#                                                                                                model_params,
#                                                                                                sample_space, k)
#     elif solver == ICON_SCHEDULING_EASY:
#         sampled_profits, sampled_predicted_profits, __ = compute_sampled_alpha_icon_energy(benchmark_X, benchmark_Y,
#                                                                                            opt_params,
#                                                                                            model_params,
#                                                                                            sample_space, k)
#
#     return sampled_profits, sampled_predicted_profits


def get_optimization_objective(X, Y, weights, model_params, opt_params):
    solver = opt_params.get('solver')
    if solver == KNAPSACK:
        return compute_profit_knapsack(X, Y, weights, model_params, opt_params)
    elif solver == ICON_SCHEDULING_EASY:
        return compute_icon_energy_cost(X, Y, weights, model_params, opt_params)
    else:
        print('error')


def get_optimal_average_objective_value(X, Y, weights, opt_params, solver='knapsack'):
    solver = opt_params.get('solver')
    if solver == KNAPSACK:
        return compute_optimal_average_value_knapsack(Y, weights, opt_params)
    elif solver == ICON_SCHEDULING_EASY:
        return compute_optimal_average_value_icon_energy(Y, opt_params)
    else:
        print('error')
