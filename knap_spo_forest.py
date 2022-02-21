from SPOTree_knapsack.SPOForest_knapsack import SPOTree_knapsack_wrapper

"""
SPO-Forest knapsack benchmarks
Dependencies:
gcc/8.3.0
openmpi/3.1.4
python/3.7.4
scikit-learn/0.23.1-python-3.7.4
gurobi/9.0.0
numpy/1.17.3-python-3.7.4

"""

# 1000 20 50 0 10 "20-50-100" "20" "100" "2-5" "mean" "SPO" 8

capacities = [12,24,48,72,96,120,144,172,196,220]
kfolds = [0,1,2,3,4]
for i in range(10):
    for c in capacities:
        for kfold in kfolds:
            SPOTree_knapsack_wrapper(max_depth_set_str="1000", min_samples_leaf_set_str="20",
                                 n_estimators_set_str="50", max_features_set_str="2-3-4-5", algtype="SPO", core_number=8,
                                 decision_problem_seed=-1, train_size=-1, quant_discret=0.05,unit_weight=False,capacity=c, kfold=kfold)


capacities = [5,10,15,20,25,30,35,40,45]
kfolds = [0, 1, 2, 3, 4]
for i in range(10):
    for c in capacities:
        for kfold in kfolds:
            SPOTree_knapsack_wrapper(max_depth_set_str="1000", min_samples_leaf_set_str="20",
                                     n_estimators_set_str="50", max_features_set_str="2-3-4-5", algtype="SPO",
                                     core_number=8,
                                     decision_problem_seed=-1, train_size=-1, quant_discret=0.05, unit_weight=True,
                                     capacity=c, kfold=kfold)

