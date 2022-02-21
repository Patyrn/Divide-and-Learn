from dnl.IconEasySolver import ICON_scheduling, calculate_energy_from_solver, ICON_scheduling_relaxation
from KnapsackSolving import solveKnapsackProblem, solveKnapsackProblemRelaxation
from dnl.Params import KNAPSACK, ICON_SCHEDULING_EASY
import numpy as np

def get_relax_optimization_objective(Y, weights, opt_params,relaxation=False):
    solver = opt_params.get('solver')
    if solver == KNAPSACK:
        return compute_obj_knapsack(Y, weights, opt_params,relaxation)
    elif solver == ICON_SCHEDULING_EASY:
        return compute_icon_scheduling_obj(Y,  opt_params,relaxation)
    else:
        print('error')


def compute_obj_knapsack( Y, weights, opt_params,relaxation):
    capacity = opt_params.get('capacity')
    obj_vals = np.zeros(len(Y))
    index = range(len(Y))
    for i, benchmark_Y, benchmark_weights in zip(index, Y, weights):
        benchmark_Y = benchmark_Y.reshape(-1).tolist()
        benchmark_weights = benchmark_weights.reshape(-1).tolist()
        if relaxation:
            solution = solveKnapsackProblemRelaxation(benchmark_Y, [benchmark_weights], capacity, warmstart=None)
        else:
            solution = solveKnapsackProblem(benchmark_Y, [benchmark_weights], capacity, warmstart=None)
        predicted_opt_items = np.asarray(solution['assignments'])
        obj_vals[i] = np.sum(benchmark_Y * predicted_opt_items)
    return obj_vals


def compute_icon_scheduling_obj(Y, opt_params,relaxation):
    obj_vals = np.zeros(len(Y))
    index = range(len(Y))
    for i, benchmark_Y in zip(index,Y):
        benchmark_Y = benchmark_Y.reshape(-1).tolist()
        if relaxation:
            solver = ICON_scheduling_relaxation(price=benchmark_Y, opt_params=opt_params, verbose=False)
            # print('Relaxation')
        else:
            objVal, solver = ICON_scheduling(price=benchmark_Y, opt_params=opt_params)
        optimal_objective_value = calculate_energy_from_solver(solver, benchmark_Y)
        obj_vals[i] = optimal_objective_value
    return obj_vals