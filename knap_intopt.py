from Experiments import test_intopt_knapsack, test_intopt_unit_knapsack

capacities = [12,24,48,72,96,120]
kfolds = [0,1,2,3,4]
dest_folder = 'Tests/intopt'
for i in range(5):
    for c in capacities:

        test_intopt_knapsack(capacity=c, is_shuffle=True, kfolds=kfolds, n_iter=1, epoch_limit=8, dest_folder=dest_folder)

