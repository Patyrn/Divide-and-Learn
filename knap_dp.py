from Experiments import test_knapsack_unit_DP,test_knapsack_weighted_DP

"""
DP knapsack experiments
Dependencies:
gcc/8.3.0
openmpi/3.1.4
python/3.7.4
scikit-learn/0.23.1-python-3.7.4
gurobi/9.0.0
numpy/1.17.3-python-3.7.4
matplotlib/3.2.1-python-3.7.4
pytorch/1.5.1-python-3.7.4

"""

capacities = [12]
test_knapsack_weighted_DP(capacities=capacities,
                          kfolds=[0, 1, 2, 3, 4],
                          core_number=8, is_shuffle=False,
                          n_iter=5, noise_level=0)

capacities = [5,10,15,20,25,30,35,40]
test_knapsack_unit_DP(capacities=capacities,
                      kfolds=[0, 1, 2, 3, 4],
                      core_number=8, is_shuffle=False,
                      n_iter=5, noise_level=0)