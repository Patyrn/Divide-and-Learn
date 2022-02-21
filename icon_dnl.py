import os


"""
This is an example to run an instance of dnl framework for Icon scheduling benchmarks.
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
from Experiments import test_Icon_unit

if __name__ == '__main__':
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_folder_greedy = os.path.join(dir_path, 'Tests/dnl_large')
        loads = [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,50,51,52,53,54,55,56,57,400,500,501]
        kfolds = [0,1,2,3,4]
        for n in range(10):
            test_Icon_unit(max_step_size_magnitude=1, min_step_size_magnitude=-1, loads=loads,
                           test_boolean=[0, 0, 0, 0, 1], core_number=8, is_shuffle=True, kfolds=kfolds, n_iter=1,
                           file_folder=file_folder_greedy)
