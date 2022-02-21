'''
Generic file to set up the decision problem (i.e., optimization problem) under consideration
Must have functions: 
  get_num_decisions(): returns number of decision variables (i.e., dimension of cost vector)
  find_opt_decision(): returns for a matrix of cost vectors the corresponding optimal decisions for those cost vectors

This particular file sets up a news article recommendation decision problem
'''

# from gurobipy import *
import numpy as np
import sys

from ortools.algorithms import pywrapknapsack_solver


def get_num_decisions(item_weights=None, capacity=None):
    return 48


def find_opt_decision(values_arr, item_weights, capacity):
    objective = np.zeros(values_arr.shape[0])
    weights = np.zeros(values_arr.shape)
    capacity = [capacity]
    item_weights = item_weights.tolist()
    for index in range(values_arr.shape[0]):
            values = values_arr[index, :]
            solver = pywrapknapsack_solver.KnapsackSolver(
                pywrapknapsack_solver.KnapsackSolver.
                    KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
            values = values.tolist()
            values = [0 if i < 0 else i for i in values]
            # print(values)
            # print(item_weights)
            # print(capacity)
            values = [int(v*100) for v in values]
            solver.Init(values, [item_weights], capacity)
            computed_value = solver.Solve()
            packed_items = []
            packed_weights = []
            total_weight = 0
            for i in range(len(values)):
                if solver.BestSolutionContains(i):
                    packed_items.append(i)
                    packed_weights.append(item_weights[i])
                    total_weight += item_weights[i]
            weights[index] = onehot(packed_items,values)
            objective[index]=calculate_profit_from_items(packed_items,values)
            # objective[index,:]=onehot(packed_items,values)
    return {'weights': weights, 'objective': objective}

    # print('Total weight:', total_weight)
    # print('Packed items:', packed_items)
    # print('Packed_weights:', packed_weights)

def onehot(packed_items, y):
    x = np.zeros(len(y))
    for i in packed_items:
        x[i] = 1
    return x
def calculate_profit_from_items(items, values):
    total_value = 0
    for i in items:
        total_value = total_value + values[i]
    return total_value

    # num_constr, num_dec = A_constr.shape
    # '''input matrix p, such that each row corresponds to an instance'''
    # weights = np.zeros(p.shape)
    # objective = np.zeros(p.shape[0])
    #
    # if (p.shape[1] != num_dec):
    #     return 'Shape inconsistent, check input dimensions.'
    #
    # model = Model()
    # model.Params.outputflag = 0
    # w = model.addVars(num_dec, lb = l_constr, ub= u_constr)
    # #w = model.addVars(num_dec, lb =0)
    # model.addConstrs((quicksum(A_constr[i][j]*w[j] for j in range(num_dec)) <= b_constr[i] for i in range(num_constr)))
    # model.addConstr(quicksum(w[j] for j in range(num_dec)) == 1)
    #
    # for inst in range(p.shape[0]):
    #     #model.setObjective(quicksum(p[inst,j]*w[j] for j in range(num_dec)), GRB.MAXIMIZE)
    #     model.setObjective(quicksum(p[inst,j]*w[j] for j in range(num_dec)), GRB.MINIMIZE)
    #     model.optimize()
    #     if model.status == GRB.OPTIMAL:
    #         weights[inst,:] = np.array([w[j].X for j in range(num_dec)])
    #         objective[inst] = model.ObjVal
    #     else:
    #         print(inst, "Infeasible!")
    #         sys.exit("Decision problem infeasible")
    # # print(model.status)
    # return {'weights': weights, 'objective':objective}
    #


'''Example input'''
# np.random.seed(0)
# gen_decision_problem()
# inst_num = 10
# p = np.random.rand(inst_num,num_dec)*-1.0
# w = find_opt_decision(p)
# w = w['weights']
# print(w)
# print(p)
