from __future__ import print_function

import os
from gurobipy import *
import numpy as np
# nbMachines: number of machine
# nbTasks: number of task
# nb resources: number of resources
# MC[m][r] resource capacity of machine m for resource r
# U[f][r] resource use of task f for resource r
# D[f] duration of tasks f
# E[f] earliest start of task f
# L[f] latest end of task f
# P[f] power use of tasks f
# idle[m] idle cost of server m
# up[m] startup cost of server m
# down[m] shut-down cost of server m
# q time resolution
# timelimit in seconds
from dnl.Params import ICON_SCHEDULING_EASY
from dnl.PredictPlustOptimizeUtils import compute_C_k, compute_F_k
from dnl.Utils import read_file


# Import Python wrapper for or-tools CP-SAT solver.


def ICON_scheduling_relaxation(price, opt_params,
                               verbose=False, scheduling=False, warmstart=None, timelimit=None):
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds
    q = opt_params.get('q')
    nbMachines = opt_params.get('nr_machines')
    nbResources = opt_params.get('nr_resources')
    nbTasks = opt_params.get('nr_tasks')
    MC = opt_params.get('MC')
    U = opt_params.get('U')
    D = opt_params.get('D')
    E = opt_params.get('E')
    L = opt_params.get('L')
    P = opt_params.get('P')
    idle = opt_params.get('idle')
    up = opt_params.get('up')
    down = opt_params.get('down')

    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440 // q
    price = np.array(price).repeat(int(30 / q))

    M = Model("icon")
    if not verbose:
        M.setParam('OutputFlag', 0)
    lb = 0.0
    ub = 1.0
    x = {}
    for f in Tasks:
        for m in Machines:
            for t in range(N):
                x[(f, m, t)] = M.addVar(lb, ub, name="x" + str(f) + "_" + str(m) + "_" + str(t))
                if warmstart is not None:
                    x[(f, m, t)].start = warmstart[f, m, t]

    # earliest start time constraint
    M.addConstrs((x[(f, m, t)] == 0) for f in Tasks for m in Machines for t in range(E[f]))
    # latest end time constraint
    M.addConstrs((x[(f, m, t)] == 0) for f in Tasks for m in Machines for t in range(L[f] - D[f] + 1, N))

    M.addConstrs((quicksum(x[(f, m, t)] for t in range(N) for m in Machines) == 1 for f in Tasks))
    # capacity requirement
    for r in Resources:
        for m in Machines:
            for t in range(N):
                M.addConstr(sum(sum(x[(f, m, t1)] for t1 in range(max(0, t - D[f] + 1), t + 1)) *
                                U[f][r] for f in Tasks) <= MC[m][r])

    print(price)
    M.setObjective(sum((x[(f, m, t)] * P[f] * sum([price[t + i] for i in range(D[f])]) * q / 60) for f in Tasks
                       for m in Machines for t in range(N - D[f] + 1)), GRB.MINIMIZE)

    if timelimit:
        M.setParam('TimeLimit', timelimit)
    M.optimize()
    schedule = {}
    if M.status == GRB.Status.OPTIMAL:
        task_on = M.getAttr('x', x)

        if verbose:
            for k, val in task_on.items():
                if int(val) > 0:
                    print("Task_%d starts on machine_%d at %d" % (k[0], k[1], k[2]))
            print('\nCost: %g' % M.objVal)
        solver = np.zeros(N)
        '''
        for t in range(N+1):
        solver[t] = sum(task[(f,m,t)]*P[f] for f in Tasks for m in Machines)*q/60 + \
        sum( machine_run[(m,t)]*idle[m] for m in  Machines)*q/60 + \
        sum(m_on[(m,t)]*up[m] for m in Machines)/price[t] + \
        sum(m_off[(m,t)]*down[m] for m in Machines)/price[t]
        '''
        for t in range(N):
            solver[t] = sum(sum(task_on[(f, m, t1)] for t1 in range(max(0, t - D[f] + 1), t + 1)) * P[f]
                            for f in Tasks for m in Machines)
        solver = solver * q / 60

        if scheduling:
            solver = np.zeros((nbTasks, nbMachines, N))
            for f in Tasks:
                for m in Machines:
                    for t in range(N):
                        solver[f, m, t] = task_on[(f, m, t)]
            return solver, schedule
        return solver
    elif M.status == 9:
        try:
            task_on = M.getAttr('x', x)
            solver = np.zeros(N)
            for t in range(N):
                solver[t] = sum(sum(task_on[(f, m, t1)] for t1 in range(max(0, t - D[f] + 1), t + 1)) * P[f]
                                for f in Tasks for m in Machines)
            solver = solver * q / 60
            if scheduling:
                schedule = np.zeros((nbTasks, nbMachines, N))
                for f in Tasks:
                    for m in Machines:
                        for t in range(N):
                            schedule[f, m, t] = task_on[(f, m, t)]
                return solver, schedule
            return solver
        except:
            print("__________Something went wrong_______")

    elif M.status == GRB.Status.INF_OR_UNBD:
        print('Model is infeasible or unbounded')

    elif M.status == GRB.Status.INFEASIBLE:
        print('Model is infeasible')
    elif M.status == GRB.Status.UNBOUNDED:
        print('Model is unbounded')

    else:
        print('Optimization ended with status %d' % M.status)

    return None


def ICON_scheduling(price, opt_params,
                    verbose=False, scheduling=False, warmstart=None, timelimit=None):
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds
    q = opt_params.get('q')
    nbMachines = opt_params.get('nr_machines')
    nbResources = opt_params.get('nr_resources')
    nbTasks = opt_params.get('nr_tasks')
    MC = opt_params.get('MC')
    U = opt_params.get('U')
    D = opt_params.get('D')
    E = opt_params.get('E')
    L = opt_params.get('L')
    P = opt_params.get('P')
    idle = opt_params.get('idle')
    up = opt_params.get('up')
    down = opt_params.get('down')

    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440 // q
    price = np.array(price).repeat(int(30 / q))
    # price = [0 if i < 0 else i for i in price]
    M = Model("icon")
    if not verbose:
        M.setParam('OutputFlag', 0)

    x = {}
    for f in Tasks:
        for m in Machines:
            for t in range(N):
                x[(f, m, t)] = M.addVar(vtype=GRB.BINARY, name="x" + str(f) + "_" + str(m) + "_" + str(t))
                if warmstart is not None:
                    x[(f, m, t)].start = warmstart[f, m, t]

    # earliest start time constraint
    M.addConstrs((x[(f, m, t)] == 0) for f in Tasks for m in Machines for t in range(E[f]))
    # latest end time constraint
    M.addConstrs((x[(f, m, t)] == 0) for f in Tasks for m in Machines for t in range(L[f] - D[f] + 1, N))
    M.addConstrs((quicksum(x[(f, m, t)] for t in range(N) for m in Machines) == 1 for f in Tasks))
    # capacity requirement
    for r in Resources:
        for m in Machines:
            for t in range(N):
                M.addConstr(sum(sum(x[(f, m, t1)] for t1 in range(max(0, t - D[f] + 1), t + 1)) *
                                U[f][r] for f in Tasks) <= MC[m][r])
    M.setObjective(sum((x[(f, m, t)] * P[f] * sum([price[t + i] for i in range(D[f])]) * q / 60) for f in Tasks
                       for m in Machines for t in range(N - D[f] + 1)), GRB.MAXIMIZE)
    if timelimit:
        M.setParam('TimeLimit', timelimit)
    M.optimize()
    if M.status == GRB.Status.OPTIMAL:
        task_on = M.getAttr('x', x)
        if verbose:
            for k, val in task_on.items():
                if int(val) > 0:
                    print("Task_%d starts on machine_%d at %d" % (k[0], k[1], k[2]))
            print('\nCost: %g' % M.objVal)
            print('\nExecution Time: %f' % M.Runtime)

        solver = np.zeros(N)
        '''
        for t in range(N+1):
        solver[t] = sum(task[(f,m,t)]*P[f] for f in Tasks for m in Machines)*q/60 + \
        sum( machine_run[(m,t)]*idle[m] for m in  Machines)*q/60 + \
        sum(m_on[(m,t)]*up[m] for m in Machines)/price[t] + \
        sum(m_off[(m,t)]*down[m] for m in Machines)/price[t]
        '''
        for t in range(N):
            solver[t] = sum(sum(task_on[(f, m, t1)] for t1 in range(max(0, t - D[f] + 1), t + 1)) * P[f]
                            for f in Tasks for m in Machines)
        solver = solver * q / 60
        if scheduling:
            schedule = np.zeros((nbTasks, nbMachines, N))
            for f in Tasks:
                for m in Machines:
                    for t in range(N):
                        schedule[f, m, t] = task_on[(f, m, t)]
            return solver, schedule
        return M.objVal, solver
    elif M.status == GRB.Status.INF_OR_UNBD:
        print('Model is infeasible or unbounded')

    elif M.status == GRB.Status.INFEASIBLE:
        print('Model is infeasible')
    elif M.status == GRB.Status.UNBOUNDED:
        print('Model is unbounded')
    elif M.status == 9:
        try:
            task_on = M.getAttr('x', x)
            solver = np.zeros(N)
            for t in range(N):
                solver = sum(sum(task_on[(f, m, t1)] for t1 in range(max(0, t - D[f] + 1), t + 1)) * P[f]
                             for f in Tasks for m in Machines for t in range(N))

            solver = solver * q / 60
            if scheduling:
                schedule = np.zeros((nbTasks, nbMachines, N))
                for f in Tasks:
                    for m in Machines:
                        for t in range(N):
                            schedule[f, m, t] = task_on[(f, m, t)]
                return solver, schedule
            return solver
        except:
            print("__________Something went wrong_______")
    else:
        print('Optimization ended with status %d' % M.status)
    return None


def compute_icon_energy_cost(X, Y, weights, model_params, opt_params):
    """
    Computes the average profit of a benchmark set
    :param const:
    :param weights:
    :param alphas: parameters of the model
    :param train_X: test set features
    :param train_Y: test set profits
    :param capacities: capacity of the problem
    :return: average_profit:
    """
    alphas = model_params.get('alphas')
    const = model_params.get('const')

    average_cost = np.zeros(len(Y))
    index = range(len(Y))

    for i, benchmark_X, benchmark_Y, benchmark_weights in zip(index, X, Y, weights):
        average_cost[i] = compute_icon_static_alphas(benchmark_X=benchmark_X, benchmark_Y=benchmark_Y,
                                                     opt_params=opt_params, alphas=alphas, const=const)
    return average_cost


def compute_sampled_alpha_icon_energy(benchmark_X, benchmark_Y, opt_params, model_params,
                                      sample_space,
                                      k):
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

    sampled_energy_consumption = []
    sampled_predicted_energy_consumption = []
    sampled_item_sets = []
    for tmp_alpha in sample_space:
        C_k = compute_C_k(benchmark_X, alphas, const, k)
        # Compute predicted cost F_k for each alpha_k in searching space for the current k
        F_k = compute_F_k(benchmark_X, tmp_alpha, C_k, k)

        # compute regret knapsack
        energy_consumption, predicted_energy_consumption, __ = compute_icon_energy_single_benchmark(F_k.reshape(1, -1),
                                                                                                    benchmark_Y,
                                                                                                    opt_params=opt_params)
        sampled_energy_consumption.append(energy_consumption)
        sampled_predicted_energy_consumption.append(predicted_energy_consumption)

    sampled_energy_consumption = np.array(sampled_energy_consumption)
    sampled_predicted_energy_consumption = np.array(sampled_predicted_energy_consumption)
    return sampled_energy_consumption, sampled_predicted_energy_consumption, sampled_item_sets


def compute_icon_static_alphas(benchmark_X, benchmark_Y, opt_params, alphas, const):
    F_k = compute_C_k(benchmark_X, alphas, const, isSampling=False)
    energy_consumption, __, __ = compute_icon_energy_single_benchmark(F_k, benchmark_Y, opt_params)
    return energy_consumption


def compute_optimal_average_value_icon_energy(Y, opt_params):
    average_cost = np.zeros(len(Y))
    index = range(len(Y))
    for i, benchmark_Y in zip(index, Y):
        benchmark_Y = benchmark_Y.reshape(-1).tolist()

        objVal, solver = ICON_scheduling(price=benchmark_Y, opt_params=opt_params)
        optimal_objective_value = calculate_energy_from_solver(solver, benchmark_Y)
        average_cost[i] = optimal_objective_value
    return average_cost


def compute_icon_energy_single_benchmark(F_k, Y, opt_params):
    """
    Computes the knapsack regret for a single benchmark. Returns both regret with respected predicted values and true values.
    Regret is defined as the loss of choosing a particular item set over the optimal item set.
    Item set choice is determined accordingly to the predicted values. However from there we have two regret calculations

    Predicted_regret: we use the item set calculated using predicted values, but when calculating cost we still use predicted item values. We propose this regret is convex therefore we can use it to find transition points.
    Regret(True regret): we use the item set calculated using predicted values, but when calculating cost we use true values. This regret is used to evaluate our algorithm.

    :param F_k: Predicted values for a alpha_k value
    :param Y: True values
    :param weights: weights of the items
    :param capacities: capacity of the knapsack problem
    :return: regret(int), predicted_regret(int)
    """
    F_k = (F_k.reshape(-1)).tolist()
    Y = Y.reshape(-1).tolist()

    objVal, solver = ICON_scheduling(price=F_k, opt_params=opt_params)
    # print('Objective Value', objVal)
    predicted_energy_consumption = calculate_energy_from_solver(solver, F_k)
    energy_consumption = calculate_energy_from_solver(solver, Y)
    # print('energy_consumption', energy_consumption)

    return energy_consumption, predicted_energy_consumption, None


def calculate_energy_from_solver(solver, price):
    # print('price', len(price))
    energy_price = np.matmul(solver, price)
    # turn minimization into maximization problem
    return energy_price


def get_icon_instance_params(instance_no=1, folder_path='data/icon_instances/easy'):
    # nr_machines: number of machine
    # nr_tasks: number of task
    # nr_resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds
    folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(folder_path, 'data/icon_instances/easy')
    filename = 'instance' + str(instance_no) + '.txt'
    raw_file = read_file(filename=filename, folder_path=folder_path)
    header_length = 2
    q = int(raw_file[0][0])
    nr_resources = int(raw_file[1][0])
    nr_machines = int(raw_file[2][0])

    MC = [[0] for resource in (range(nr_resources)) for machines in range(nr_machines)]
    for m in range(nr_machines):
        for r in range(nr_resources):
            MC[m][r] = int(raw_file[header_length + ((m + 1) * 2)][r])

    idle = [0 for machines in range(nr_machines)]
    for m in range(nr_machines):
        idle[m] = int(raw_file[header_length + 1 + (m * 2)][1])

    up = [0 for machines in range(nr_machines)]
    for m in range(nr_machines):
        up[m] = float(raw_file[header_length + 1 + (m * 2)][2])

    down = [0 for machines in range(nr_machines)]  #
    for m in range(nr_machines):
        down[m] = float(raw_file[header_length + 1 + (m * 2)][3])

    task_start_index = header_length + nr_machines * 2 + 1
    nr_tasks = int(raw_file[task_start_index][0])

    U = [[0] for resource in range(nr_resources) for task in range(nr_tasks)]
    for f in range(nr_tasks):
        for r in range(nr_resources):
            U[f][r] = int(raw_file[task_start_index + ((f + 1) * 2)][r])

    D = [0 for tasks in range(nr_tasks)]
    for f in range(nr_tasks):
        D[f] = int(raw_file[task_start_index + 1 + (f * 2)][1])

    E = [0 for tasks in range(nr_tasks)]
    for f in range(nr_tasks):
        E[f] = int(raw_file[task_start_index + 1 + (f * 2)][2])

    L = [0 for tasks in range(nr_tasks)]
    for f in range(nr_tasks):
        L[f] = int(raw_file[task_start_index + 1 + (f * 2)][3])

    P = [0 for tasks in range(nr_tasks)]
    for f in range(nr_tasks):
        P[f] = float(raw_file[task_start_index + 1 + (f * 2)][4])

    opt_params = {'q': q,
                  'nr_machines': nr_machines,
                  'nr_resources': nr_resources,
                  'nr_tasks': nr_tasks,
                  'MC': MC,
                  'U': U,
                  'D': D,
                  'E': E,
                  'L': L,
                  'P': P,
                  'idle': idle,
                  'up': up,
                  'down': down,
                  'solver': ICON_SCHEDULING_EASY
                  }
    return opt_params


if __name__ == '__main__':
    opt_params = get_icon_instance_params(2)
    icon1 = icon_solver(opt_params, time_limit=0.01)
    icon1 = icon_solver(opt_params, time_limit=1)
    icon2 = icon_solver(opt_params, time_limit=10)
    icon3 = icon_solver(opt_params, time_limit=30, verbose=False)
    icon4 = icon_solver(opt_params, time_limit=60)

    print(icon1)
    print(icon2)
    print(icon3)
    print(icon4)
