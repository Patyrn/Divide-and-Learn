from Experiments import test_QPTL
"""
Example qptl experiment for scheduling benchmarks
Dependencies:
gcc/8.3.0
openmpi/3.1.4
python/3.7.4
scikit-learn/0.23.1-python-3.7.4
gurobi/9.0.0
numpy/1.17.3-python-3.7.4
matplotlib/3.2.1-python-3.7.4
"""
if __name__ == '__main__':
    loads = [30,31,32,33]
    kfolds = [0,1,2,3,4]
    dest_folder = 'Tests/qptl/'
    for n in range(10):
        for load in loads:
            test_QPTL(instance_number=load, is_shuffle=True, kfolds=kfolds, n_iter=1, dest_folder=dest_folder)
