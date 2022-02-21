import copy
import csv
import os
import numpy as np

from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder

from dnl.Params import RANDOM_SEED


class Point:
    """
    Utility class for points
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def is_same(self, point):
        if self.x == point.x and self.y == point.y:
            return True
        else:
            return False


class TransitionPoint:
    """
    Utility class for points
    """

    def __init__(self, x=0, predicted_profit=0, true_profit=0):
        self.x = x
        self.predicted_profit = predicted_profit
        self.true_profit = true_profit

    def is_same(self, point):
        if self.x == point.x and self.y == point.y:
            return True
        else:
            return False


class Interval:
    """
    Utility class for interval
    """

    def __init__(self, starting_point=Point(-100, 0), ending_point=Point(100, 0)):
        self.starting_point = starting_point
        self.ending_point = ending_point


def create_intervals_from_points(point_list):
    intervals = []
    for i in range(len(point_list) - 1):
        starting_point = point_list[i]
        ending_point = point_list[i + 1]
        interval = Interval(starting_point=starting_point, ending_point=ending_point)
        intervals.append(interval)
    return intervals


def create_transition_points_from_intervals(interval_list, selection_method='EDGE'):
    N = len(interval_list)
    transition_points = [[] for i in range(N)]
    for i in range(N):
        for interval in interval_list[i]:
            if selection_method == 'MID_POINT':
                point = TransitionPoint((interval.ending_point.x + interval.starting_point.x) / 2,
                                        true_profit=interval.ending_point.true_profit)
                transition_points[i].append(point)

            elif selection_method == 'EDGE':
                start_point = interval.starting_point
                end_point = interval.ending_point

                if len(transition_points[i]) == 0:
                    transition_points[i].append(copy.deepcopy(start_point))
                elif not (transition_points[i][-1].x == start_point.x):
                    transition_points[i].append(copy.deepcopy(start_point))
                transition_points[i].append(copy.deepcopy(end_point))
    return transition_points


def read_file(filename, folder_path='data', delimiter=' '):
    """
    read the dataset with filename and return the feature and labels in list.
    :param filename (str): filename of the dataset
    :return: data(list) : all features and labels in the data. We transform the data into features and labels with a different method, unique for each dataset.
    """
    file_path = get_file_path(filename, folder_path=folder_path)

    with open(file_path, 'r') as f:
        data = list(csv.reader(f, delimiter=delimiter))
    return data


def get_file_path(filename, folder_path='data'):
    """
    Constructs filepath. dataset is expected to be in the "data" folder
    :param filename:
    :return:
    """
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(dir_path, folder_path, filename)
    return file_path


def get_train_test_split(dataset, random_seed=RANDOM_SEED, is_shuffle=False):
    """
    dataset is already seperated into benchmarks, split data but preserve benchmarks.
    Splits the dataset into train and test sets. Also constructs the weight vector. currently we use unit weight but it can be changed in the future.
    :param X: Features
    :param Y: Outputs
    :return: train_set(dictionary), test_set(dictionary)
    """
    benchmarks_X = dataset.get('benchmarks_X')
    benchmarks_Y = dataset.get('benchmarks_Y')
    benchmarks_weights = dataset.get('benchmarks_weights')

    number_of_benchmarks = len(benchmarks_X)
    benchmark_indexes = np.arange(number_of_benchmarks)

    benchmarks_X_train_index, benchmarks_X_test_index, benchmarks_Y_train_index, benchmarks_Y_test_index = model_selection.train_test_split(
        benchmark_indexes, benchmark_indexes, test_size=0.2, shuffle=is_shuffle,
        random_state=random_seed)

    benchmarks_X_train = [benchmarks_X[index] for index in benchmarks_X_train_index]
    benchmarks_Y_train = [benchmarks_Y[index] for index in benchmarks_Y_train_index]
    benchmarks_weights_train = [benchmarks_weights[index] for index in benchmarks_X_train_index]

    benchmarks_X_test = [benchmarks_X[index] for index in benchmarks_X_test_index]
    benchmarks_Y_test = [benchmarks_Y[index] for index in benchmarks_Y_test_index]
    benchmarks_weights_test = [benchmarks_weights[index] for index in benchmarks_Y_test_index]

    # change weights if neccesary, energy data does not have weights
    X_train = flatten_benchmarks(benchmarks_X_train)
    Y_train = flatten_benchmarks(benchmarks_Y_train)
    weights_train = flatten_benchmarks(benchmarks_weights_train)

    X_test = flatten_benchmarks(benchmarks_X_test)
    Y_test = flatten_benchmarks(benchmarks_Y_test)
    weights_test = flatten_benchmarks(benchmarks_weights_test)
    train_set = {'X': X_train.T,
                 'Y': Y_train.T,
                 'weights': weights_train,
                 'benchmarks_X': benchmarks_X_train,
                 'benchmarks_Y': benchmarks_Y_train,
                 'benchmarks_weights': benchmarks_weights_train}
    test_set = {'X': X_test.T,
                'Y': Y_test.T,
                'weights': weights_test,
                'benchmarks_X': benchmarks_X_test,
                'benchmarks_Y': benchmarks_Y_test,
                'benchmarks_weights': benchmarks_weights_test}
    return train_set, test_set


def get_train_test_split_SPO(dataset, random_seed=RANDOM_SEED, is_shuffle=False):
    """
    dataset is already seperated into benchmarks, split data but preserve benchmarks.
    Splits the dataset into train and test sets. Also constructs the weight vector. currently we use unit weight but it can be changed in the future.
    :param X: Features
    :param Y: Outputs
    :return: train_set(dictionary), test_set(dictionary)
    """
    benchmarks_X = dataset.get('benchmarks_X')
    benchmarks_Y = dataset.get('benchmarks_Y')
    benchmarks_weights = dataset.get('benchmarks_weights')

    number_of_benchmarks = len(benchmarks_X)
    benchmark_indexes = np.arange(number_of_benchmarks)

    benchmarks_X_train_index, benchmarks_X_test_index, benchmarks_Y_train_index, benchmarks_Y_test_index = model_selection.train_test_split(
        benchmark_indexes, benchmark_indexes, test_size=0.2, shuffle=is_shuffle,
        random_state=random_seed)

    starting_index = 0
    benchmarks_X_train = [np.vstack((np.ones(48) * starting_index + ind, benchmarks_X[index])) for ind, index in
                          enumerate(benchmarks_X_train_index)]
    benchmarks_Y_train = [benchmarks_Y[index] for index in benchmarks_Y_train_index]
    benchmarks_weights_train = [benchmarks_weights[index] for index in benchmarks_X_train_index]

    starting_index = len(benchmarks_X_train)
    print('starting index', starting_index)

    benchmarks_X_test = [np.vstack((np.ones(48) * starting_index + ind, benchmarks_X[index])) for ind, index in
                         enumerate(benchmarks_X_test_index)]
    benchmarks_Y_test = [benchmarks_Y[index] for index in benchmarks_Y_test_index]
    benchmarks_weights_test = [benchmarks_weights[index] for index in benchmarks_Y_test_index]

    # change weights if neccesary, energy data does not have weights
    X_train = flatten_benchmarks(benchmarks_X_train)
    Y_train = flatten_benchmarks(benchmarks_Y_train)
    weights_train = flatten_benchmarks(benchmarks_weights_train)

    X_test = flatten_benchmarks(benchmarks_X_test)
    Y_test = flatten_benchmarks(benchmarks_Y_test)
    weights_test = flatten_benchmarks(benchmarks_weights_test)
    train_set = {'X': X_train.T,
                 'Y': Y_train.flatten().T,
                 'benchmarks_X': benchmarks_X_train,
                 'benchmarks_Y': benchmarks_Y_train,
                 'benchmarks_weights': benchmarks_weights_train
                 }

    test_set = {'X': X_test.T,
                'Y': Y_test.flatten().T,
                'benchmarks_X': benchmarks_X_test,
                'benchmarks_Y': benchmarks_Y_test,
                'benchmarks_weights': benchmarks_weights_test
                }
    return train_set, test_set


def get_kfold_train_test_split(dataset, n_splits=5):
    """
    dataset is already seperated into benchmarks, split data but preserve benchmarks.
    Splits the dataset into train and test sets. Also constructs the weight vector. currently we use unit weight but it can be changed in the future.
    :param X: Features
    :param Y: Outputs
    :return: train_set(dictionary), test_set(dictionary)
    """
    benchmarks_X = dataset.get('benchmarks_X')
    benchmarks_Y = dataset.get('benchmarks_Y')
    benchmarks_weights = dataset.get('benchmarks_weights')

    number_of_benchmarks = len(benchmarks_X)
    benchmark_indexes = np.arange(number_of_benchmarks)

    benchmarks_X_train_index, benchmarks_X_test_index, benchmarks_Y_train_index, benchmarks_Y_test_index = model_selection.train_test_split(
        benchmark_indexes, benchmark_indexes, test_size=0.3,
        random_state=RANDOM_SEED)

    benchmarks_X_train = [benchmarks_X[index] for index in benchmarks_X_train_index]
    benchmarks_Y_train = [benchmarks_Y[index] for index in benchmarks_Y_train_index]
    benchmarks_weights_train = [benchmarks_weights[index] for index in benchmarks_X_train_index]

    benchmarks_X_test = [benchmarks_X[index] for index in benchmarks_X_test_index]
    benchmarks_Y_test = [benchmarks_Y[index] for index in benchmarks_Y_test_index]
    benchmarks_weights_test = [benchmarks_weights[index] for index in benchmarks_Y_test_index]

    # change weights if neccesary, energy data does not have weights
    X_train = flatten_benchmarks(benchmarks_X_train)
    Y_train = flatten_benchmarks(benchmarks_Y_train)
    weights_train = flatten_benchmarks(benchmarks_weights_train)

    X_test = flatten_benchmarks(benchmarks_X_test)
    Y_test = flatten_benchmarks(benchmarks_Y_test)
    weights_test = flatten_benchmarks(benchmarks_weights_test)
    train_set = {'X': X_train.T,
                 'Y': Y_train.T,
                 'weights': weights_train,
                 'benchmarks_X': benchmarks_X_train,
                 'benchmarks_Y': benchmarks_Y_train,
                 'benchmarks_weights': benchmarks_weights_train}
    test_set = {'X': X_test.T,
                'Y': Y_test.T,
                'weights': weights_test,
                'benchmarks_X': benchmarks_X_test,
                'benchmarks_Y': benchmarks_Y_test,
                'benchmarks_weights': benchmarks_weights_test}
    return train_set, test_set


def flatten_benchmarks(benchmarks):
    """

    :param benchmarks: a list of np arrays
    :return:
    """
    number_of_benchmarks = len(benchmarks)
    print(benchmarks[0].shape)
    number_of_features, size_of_a_benchmark = benchmarks[0].shape

    flattened_np_array = np.ones((number_of_features, number_of_benchmarks * size_of_a_benchmark))
    for i in range(number_of_benchmarks):
        starting_index = i * size_of_a_benchmark
        flattened_np_array[:, starting_index:starting_index + size_of_a_benchmark] = benchmarks[i]
    return flattened_np_array

def one_hot_timeslot(len):
     problem_set = np.array([x for x in range(48)] * len)
     # binary encode
     b = np.zeros((problem_set.size, problem_set.max() + 1))
     b[np.arange(problem_set.size), problem_set] = 1

     return b.T
def get_benchmarks(X, Y, benchmark_size, generate_weight=True, unit_weight=True, weight_seed=None, add_weights=False,noise_level=0):
    """
    Splits the dataset into benchmarks of a certain size for the optimization problem. Used in the second stage.
    Might not be used in the feature if we choose to use predetermined benchmarks.
    :param is_weighted:
    :param X:
    :param Y:
    :param benchmark_size:
    :return:
    """

    #Add time slot number
    # time_slots = np.array(list(np.array([x for x in range(48)])) * int(Y.shape[1]/48))
    # time_slots_rev = np.array(list(np.array([x for x in reversed(range(48))])) * int(Y.shape[1] / 48))
    # time_slots = one_hot_timeslot(int(Y.shape[1]/48))
    # X = np.vstack((X, time_slots))
    # X = np.vstack((X, time_slots_rev.reshape(1, -1)))

    feature_size, sample_size = X.shape
    if weight_seed is not None:
        if unit_weight:
            weight_seed = [1]
        else:
            weight_seed = [3, 5, 7]
            # IMPLEMENT ADDING NOISE
            np.random.seed(RANDOM_SEED)
            # if noise_level > 0:
            #     print('noise generate')
            #     noise = (1-np.random.random(sample_size)*noise_level/100)
            #     Y = Y * noise

    benchmark_count = int(sample_size / benchmark_size)
    benchmarks_X = []
    benchmarks_Y = []
    benchmarks_weights = []

    # do weights if needed
    if generate_weight:
        weights = np.ones(Y.shape)
        noisy_weights =  np.ones(Y.shape)
    else:
        weights = X[-1, :].reshape(Y.shape)
    for i in range(benchmark_count):
        start_index = i * benchmark_size
        end_index = start_index + benchmark_size

        benchmark_X = X[1:, start_index:end_index].reshape(feature_size - 1, benchmark_size)

        if generate_weight:
            if unit_weight:
                benchmark_weights = generate_uniform_weights_from_seed(benchmark_size, weight_seed)
                benchmark_noisy_weights = benchmark_weights
            else:
                #set same weight array for SPO implementation
                benchmark_weights = np.array(
                    [5, 3, 3, 5, 5, 7, 7, 3, 7, 7, 3, 3, 5, 3, 7, 3, 7, 7, 5, 5, 3, 5, 5, 3, 7, 7, 3, 7, 5, 5, 7, 3, 7,
                     3,
                     3, 5, 7, 5, 3, 5, 3, 7, 5, 7, 5, 5, 3, 7]).reshape(1, 48)
                benchmark_noisy_weights = benchmark_weights+(np.ones(benchmark_weights.shape)*noise_level)
            # benchmark_weights = np.hstack([seed_array for i in range(int(sample_size/len(seed_array)))])
            Y[:, start_index:end_index] = Y[:, start_index:end_index] * benchmark_noisy_weights
            weights[:, start_index:end_index] = benchmark_weights
            noisy_weights[:, start_index:end_index] = benchmark_noisy_weights
            benchmark_X = np.vstack((benchmark_X, benchmark_noisy_weights   .flatten()))

        else:
            benchmark_weights = weights[:, start_index:end_index].reshape(1, benchmark_size)

        benchmark_Y = Y[:, start_index:end_index].reshape(1, benchmark_size)

        benchmarks_X.append(benchmark_X)
        benchmarks_Y.append(benchmark_Y)
        benchmarks_weights.append(benchmark_weights)

    if generate_weight and not unit_weight and add_weights:
        X = np.vstack((X, benchmark_noisy_weights.flatten()))


    X = np.delete(X, 0, 0)

    dataset = {'X': X,
               'Y': Y,
               'weights': weights,
               'benchmarks_X': benchmarks_X,
               'benchmarks_Y': benchmarks_Y,
               'benchmarks_weights': benchmarks_weights}
    return dataset




def get_train_test_split_spotree(dataset, random_seed=RANDOM_SEED, is_shuffle=False):
    """
    dataset is already seperated into benchmarks, split data but preserve benchmarks.
    Splits the dataset into train and test sets. Also constructs the weight vector. currently we use unit weight but it can be changed in the future.
    :param X: Features
    :param Y: Outputs
    :return: train_set(dictionary), test_set(dictionary)
    """
    benchmarks_X = dataset.get('benchmarks_X')
    benchmarks_Y = dataset.get('benchmarks_Y')
    benchmarks_weights = dataset.get('benchmarks_weights')

    number_of_benchmarks = len(benchmarks_X)
    benchmark_indexes = np.arange(number_of_benchmarks)

    benchmarks_X_train_index, benchmarks_X_test_index, benchmarks_Y_train_index, benchmarks_Y_test_index = model_selection.train_test_split(
        benchmark_indexes, benchmark_indexes, test_size=0.2, shuffle=is_shuffle,
        random_state=random_seed)

    benchmarks_X_train = [benchmarks_X[index] for index in benchmarks_X_train_index]
    benchmarks_Y_train = [benchmarks_Y[index].reshape(-1) for index in benchmarks_Y_train_index]
    benchmarks_weights_train = [benchmarks_weights[index].reshape(-1) for index in benchmarks_X_train_index]

    benchmarks_X_test = [benchmarks_X[index] for index in benchmarks_X_test_index]
    benchmarks_Y_test = [benchmarks_Y[index].reshape(-1) for index in benchmarks_Y_test_index]
    benchmarks_weights_test = [benchmarks_weights[index].reshape(-1) for index in benchmarks_Y_test_index]

    # change weights if neccesary, energy data does not have weights
    # X_train = flatten_benchmarks(benchmarks_X_train)
    # Y_train = flatten_benchmarks(benchmarks_Y_train)
    # weights_train = flatten_benchmarks(benchmarks_weights_train)

    # X_test = flatten_benchmarks(benchmarks_X_test)
    # Y_test = flatten_benchmarks(benchmarks_Y_test)
    # weights_test = flatten_benchmarks(benchmarks_weights_test)
    train_set = {
                 # 'X': X_train.T,
                 # 'Y': Y_train.T,
                 # 'weights': weights_train,
                 'benchmarks_X': np.array(benchmarks_X_train),
                 'benchmarks_Y': np.array(benchmarks_Y_train),
                 'benchmarks_weights': np.array(benchmarks_weights_train)}
    test_set = {
                # 'X': X_test.T,
                # 'Y': Y_test.T,
                # 'weights': weights_test,
                'benchmarks_X': np.array(benchmarks_X_test),
                'benchmarks_Y': np.array(benchmarks_Y_test),
                'benchmarks_weights': np.array(benchmarks_weights_test)}
    return train_set, test_set


def get_benchmarks_spotree(X, Y, benchmark_size, generate_weight=True, unit_weight=True, weight_seed=None, add_weights=True,noise_level=0):
    """
    Splits the dataset into benchmarks of a certain size for the optimization problem. Used in the second stage.
    Might not be used in the feature if we choose to use predetermined benchmarks.
    :param is_weighted:
    :param X:
    :param Y:
    :param benchmark_size:
    :return:
    """

    feature_size, sample_size = X.shape
    if weight_seed is not None:
        if unit_weight:
            weight_seed = [1]
        else:
            weight_seed = [3, 5, 7]
            # IMPLEMENT ADDING NOISE
            np.random.seed(RANDOM_SEED)
            # if noise_level > 0:
            #     print('noise generate')
            #     noise = (1-np.random.random(sample_size)*noise_level/100)
            #     Y = Y * noise

    benchmark_count = int(sample_size / benchmark_size)
    benchmarks_X = []
    benchmarks_Y = []
    benchmarks_weights = []

    # do weights if needed
    if generate_weight:
        weights = np.ones(Y.shape)
        noisy_weights =  np.ones(Y.shape)
    else:
        weights = X[-1, :].reshape(Y.shape)
    for i in range(benchmark_count):
        start_index = i * benchmark_size
        end_index = start_index + benchmark_size

        benchmark_X = X[1:, start_index:end_index].reshape(feature_size - 1, benchmark_size)
        if generate_weight:
            if unit_weight:
                benchmark_weights = generate_uniform_weights_from_seed(benchmark_size, weight_seed).astype(int)
                benchmark_noisy_weights = benchmark_weights
            else:
                #set same weight array for SPO implementation
                benchmark_weights = np.array(
                    [5, 3, 3, 5, 5, 7, 7, 3, 7, 7, 3, 3, 5, 3, 7, 3, 7, 7, 5, 5, 3, 5, 5, 3, 7, 7, 3, 7, 5, 5, 7, 3, 7,
                     3,
                     3, 5, 7, 5, 3, 5, 3, 7, 5, 7, 5, 5, 3, 7]).reshape(1, 48)
                benchmark_noisy_weights = benchmark_weights+(np.ones(benchmark_weights.shape)*noise_level)
            # benchmark_weights = np.hstack([seed_array for i in range(int(sample_size/len(seed_array)))])
            Y[:, start_index:end_index] = Y[:, start_index:end_index] * benchmark_noisy_weights
            weights[:, start_index:end_index] = benchmark_weights
            noisy_weights[:, start_index:end_index] = benchmark_noisy_weights
            benchmark_X = np.vstack((benchmark_X, benchmark_noisy_weights   .flatten()))
            # benchmark_X = np.mean(benchmark_X,axis=1)

        else:
            benchmark_weights = weights[:, start_index:end_index].reshape(1, benchmark_size)

        benchmark_Y = Y[:, start_index:end_index].reshape(1, benchmark_size)

        benchmarks_X.append(benchmark_X.T.flatten())
        benchmarks_Y.append(benchmark_Y)
        benchmarks_weights.append(benchmark_weights)

    # if generate_weight and not unit_weight and add_weights:
    #     X = np.vstack((X, benchmark_noisy_weights.flatten()))
    X = np.delete(X, 0, 0)

    dataset = {'X': X,
               'Y': Y,
               'weights': weights,
               'benchmarks_X': benchmarks_X,
               'benchmarks_Y': benchmarks_Y,
               'benchmarks_weights': benchmarks_weights}
    return dataset



def generate_uniform_weights_from_seed(benchmark_size, weight_seed):
    number_of_each_weight = int(benchmark_size / len(weight_seed))
    uniform_weights_from_seed = np.array(
        [np.ones((number_of_each_weight)) * weight for weight in weight_seed]).flatten()
    np.random.shuffle(uniform_weights_from_seed)
    uniform_weights_from_seed = uniform_weights_from_seed.reshape((1, uniform_weights_from_seed.size))

    return uniform_weights_from_seed


def get_mini_batches(X, Y, weights, size=32):
    number_of_minibatches = int(len(X) / size)
    mini_batch_X = [] * number_of_minibatches
    mini_batch_Y = [] * number_of_minibatches
    mini_batch_weights = [] * number_of_minibatches
    # for i in a:
    #     start_i = size*i
    #     end_i = start_i + size
    #     mini
    #     mini_batch_Y.append(Y[start_i:end_i])
    #     mini_batch_weights.append(weights[start_i:end_i])

    for start_index in range(number_of_minibatches):
        mini_batch_X.append([X[j] for j in range(start_index * size, (start_index + 1) * size)])
        mini_batch_Y.append([Y[j] for j in range(start_index * size, (start_index + 1) * size)])
        mini_batch_weights.append([weights[j] for j in range(start_index * size, (start_index + 1) * size)])

    return mini_batch_X, mini_batch_Y, mini_batch_weights


def save_results_dict(file_name, file_folder, results):
    file_path = get_file_path(filename=file_name, folder_path=file_folder)
    with open(file_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')

        for key, value in results.items():
            csvwriter.writerow(value)


def save_results_list(file_name, file_folder, results):
    file_path = get_file_path(filename=file_name, folder_path=file_folder)
    with open(file_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')

        for row in results:
            csvwriter.writerow(row)
