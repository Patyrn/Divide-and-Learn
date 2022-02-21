


from SPOTree_scheduling.SPOForest_scheduling import SPOTree_scheduling
"""
Example SPO-Forest experiment for scheduling benchmarks.
Dependencies:
gcc/8.3.0
openmpi/3.1.4
python/3.7.4
scikit-learn/0.23.1-python-3.7.4
gurobi/9.0.0
numpy/1.17.3-python-3.7.4
"""

loads = [30, 31, 32, 33, 34, 35, 36]
kfolds = [1,2,3,4]
for i in range(10):
    for load in loads:
        for kfold in kfolds:
            SPOTree_scheduling(max_depth_set_str="1000", min_samples_leaf_set_str="20",
                                 n_estimators_set_str="50", max_features_set_str="2-3-4-5", algtype="SPO", core_number=8,
                                 decision_problem_seed=-1, train_size=-1, quant_discret=0.05,load=load,kfold=kfold)