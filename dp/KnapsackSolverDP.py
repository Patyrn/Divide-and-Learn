from dnl.Params import KNAPSACK
from dp.PierceWiseLinearFunction import PiercewiseLinearFunction, maxPiercewiseLinearFunctions, LinearFunction, \
    addPiercewiseLinearFunctions, convertLinFunctionToPLF


def KnapsackSolverDP(C, weights, values):
    n = len(values)
    C = C[0]
    weights = weights.flatten().astype(int)


    knapsack = [[0 for x in range(C + 1)] for x in range(n + 1)]
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(C + 1):
            if i == 0 or w == 0:
                knapsack[i][w] = PiercewiseLinearFunction()
            elif weights[i - 1] <= w:
                knapsack[i][w] = maxPiercewiseLinearFunctions(addPiercewiseLinearFunctions(convertLinFunctionToPLF(values[i - 1])
                              , knapsack[i - 1][w - weights[i - 1]]), knapsack[i - 1][w])
            else:
                knapsack[i][w] = knapsack[i - 1][w]

    return knapsack[n][C]

def get_opt_params_knapsack_DP(capacity=24):
    params = {'capacity': [capacity],
              'solver': KNAPSACK}
    return params

if __name__ == '__main__':
    print("main")
    # Driver program to test above function
    slopes = [-1, 0, 1]
    constants = [3,1,1]
    val_func = []
    for slope,constant in zip(slopes,constants):
        val_func.append(LinearFunction(slope=slope,constant=constant))
    wt = [1, 1, 1]
    W = 2
    a = KnapsackSolverDP(W, wt, val_func)
    print(a)