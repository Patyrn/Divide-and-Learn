from Experiments import test_knapsack_SPO_unit, test_knapsack_SPO

"""
Example SPO-Relax experiments for knapsack benchmarks.
Dependencies:
gcc/8.3.0
openmpi/3.1.4
python/3.7.4
scikit-learn/0.23.1-python-3.7.4
gurobi/9.0.0
numpy/1.17.3-python-3.7.4
matplotlib/3.2.1-python-3.7.4
"""
capacities = [12,24,48,72,96,120,144,172,196,220]
kfolds = [0,1,2,3,4]


dest_folder = 'Tests/Knapsack/weighted/spo'
test_knapsack_SPO(capacities=capacities, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=kfolds, n_iter=5,
                  dest_folder=dest_folder, noise_level=0)
dest_folder = 'Tests/Knapsack/unit/spo'
capacities = [5,10,15,20,25,30,35,40]
test_knapsack_SPO_unit(capacities=capacities, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=kfolds, n_iter=5,
                  dest_folder=dest_folder, noise_level=0)