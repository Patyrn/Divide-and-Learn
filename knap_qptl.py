from Experiments import test_knapsack_qptl, test_knapsack_qptl_unit
"""
QPTL knapsack experiments
Dependencies
gcc/8.3.0
openmpi/3.1.4
python/3.7.4
scikit-learn/0.23.1-python-3.7.4
gurobi/9.0.0
numpy/1.17.3-python-3.7.4
matplotlib/3.2.1-python-3.7.4
torch 1.0.0

"""
capacities = [12,24,48,72,96,120,144,172,196,220]
kfolds = [0]
dest_folder = 'Tests/Knapsack/weighted/qptl/'
test_knapsack_qptl(capacities=capacities, is_shuffle=False,
                       NUMBER_OF_RANDOM_TESTS=1, kfolds=kfolds,
                       n_iter=5, dest_folder=dest_folder, noise_level=0)

capacities = [5,10,15,20,25,30,35,40]
dest_folder = 'Tests/Knapsack/unit/qptl/'
test_knapsack_qptl_unit(capacities=capacities, is_shuffle=False,
                       NUMBER_OF_RANDOM_TESTS=1, kfolds=kfolds,
                       n_iter=5, dest_folder=dest_folder, noise_level=0)
