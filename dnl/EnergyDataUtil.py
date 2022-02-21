import os

import numpy as np

from dnl.Utils import read_file, get_benchmarks, get_benchmarks_spotree

BENCHMARK_SIZE = 48

def get_energy_data(filename, generate_weight = True, unit_weight = True,kfold=0, noise_level = 0, is_spo_tree=False):
    """
    Reads the energy dataset with the filename, splits it into feature and output sets.
    :param filename:
    :return:dataset: contains X and Y(nparray), dataset_params: contains feature_size and sample_size (int)
    """
    HEADER_LENGTH = 4
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(dir_path,'data',filename)
    data = read_file(file_path)
    dataset = transform_energy_data(data, HEADER_LENGTH, generate_weight, unit_weight,kfold,noise_level=noise_level, is_spo_tree=is_spo_tree)
    return dataset

def transform_energy_data(data, header_length, generate_weight=True, unit_weight = True,kfold=0,noise_level=0,is_spo_tree=False):
    """
    transform method for energy data. Takes raw file and splits it into features and labels.
    For the energy data, first feature is actually the benchmark No
    :param data (list): List contains the data set:
    :param header_length (int): Length of header in the file. Used to decide where the itemset starts
    :return: Dataset(dictionary) holds two nparry, X: features , Y: labels. dataset_params(dictionary) contains feature_size and sample_size
    """
    weight_seed = np.array([3,5,7])
    # weights = np.random.choice(weight_seed, Y.size)
    # X = np.vstack([X, weights])
    # Y = Y * weights



    sample_size = int(data[header_length][0])
    feature_size = int(data[header_length][1])
    data = np.array(data[(header_length + 1):])

    X = data[:, 0:feature_size]
    X = np.asfarray(np.array(X), float).T
    X = X.reshape(feature_size, sample_size)

    Y = data[:, feature_size]
    Y = np.asfarray(np.array(Y), float).T
    Y = Y.reshape(1, sample_size)

    k_fold_rotation = int(48 * (750 / 5))

    X = np.roll(X,kfold*k_fold_rotation,axis=1)
    Y = np.roll(Y,kfold*k_fold_rotation,axis=1)
    # X = X[[0,7],:].reshape(2,-1)
    if is_spo_tree:
        dataset = get_benchmarks_spotree(X,Y, BENCHMARK_SIZE, generate_weight, unit_weight, weight_seed,noise_level=noise_level)
    else:
        dataset = get_benchmarks(X,Y, BENCHMARK_SIZE, generate_weight, unit_weight, weight_seed,noise_level=noise_level)

    # dataset = {'X': X,
    #            'Y': Y}
    #
    dataset['feature_size'] = feature_size - 1,
    dataset['sample_size'] = sample_size
    dataset['benchmark_size'] = BENCHMARK_SIZE
    return dataset

def get_energy_data_benchmarks(X, Y, is_weighted = False):

    sample_size = int(len(X[0]))
    feature_size = int(len(X))

    weights = np.zeros(sample_size)
    benchmark_weights = np.zeros(BENCHMARK_SIZE)
    number_of_benchmarks = int(sample_size / BENCHMARK_SIZE)
    benchmarks_X = []
    benchmarks_Y = []

def transform_energy_data_with_weights(data, header_length):
    """
    transform method for energy data. Takes raw file and splits it into features and labels.
    For the energy data, first feature is actually the benchmark No
    :param data (list): List contains the data set:
    :param header_length (int): Length of header in the file. Used to decide where the itemset starts
    :return: Dataset(dictionary) holds two nparry, X: features , Y: labels. dataset_params(dictionary) contains feature_size and sample_size
    """
    weight_seed = np.array([3,5,7])
    # weight_seed = np.array([1])

    sample_size = int(data[header_length][0])
    feature_size = int(data[header_length][1])
    data = np.array(data[(header_length + 1):])

    X = data[:, 1:feature_size]
    X = np.asfarray(np.array(X), float).T
    X = X.reshape(feature_size-1, sample_size)




    Y = data[:, feature_size]
    Y = np.asfarray(np.array(Y), float).T
    Y = Y.reshape(1, sample_size)

    weights = np.random.choice(weight_seed, Y.size)
    X = np.vstack([X, weights])

    Y = Y * weights

    dataset = {'X': X,
               'Y': Y}

    dataset_params = {'feature_size': feature_size-1,
                      'sample_size': sample_size}
    return dataset, dataset_params
