from Experiments import  test_knapsack_weighted, test_knapsack_unit

"""
Example Dnl experiments on weighted and unit knapsack problems. 
Test boolean (boolean array): determines the variations of dnl used. The order is [Exhaustive, Exhaustive_max, Dnl, dnl_max, dnl_greedy]. exhaustive, dnl and dnl_greedy are used in the paper.
for dnl_greedy choose test boolean = [0,0,0,0,1]

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



test_knapsack_weighted(max_step_size_magnitude=0, min_step_size_magnitude=-1, capacities=capacities, epoch_limit=3,
                       kfolds=kfolds,
                       test_boolean=[0, 0, 0, 0, 1], core_number=8, is_shuffle=False, learning_rate=0.1,
                       mini_batch_size=32, n_iter=5, noise_level=0)

capacities = [5,10,15,20,25,30,35,40]

test_knapsack_unit(max_step_size_magnitude=0, min_step_size_magnitude=-1, capacities=capacities, epoch_limit=3,
                       kfolds=kfolds,
                       test_boolean=[0, 0, 0, 0, 1], core_number=8, is_shuffle=False, learning_rate=0.1,
                       mini_batch_size=32, n_iter=5, noise_level=0)