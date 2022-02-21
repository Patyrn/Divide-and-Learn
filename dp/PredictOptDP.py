import copy
import random
import time
from functools import partial
from operator import attrgetter

from dnl import Solver
from dnl.EnergyDataUtil import get_energy_data
from dp.KnapsackSolverDP import KnapsackSolverDP, get_opt_params_knapsack_DP
from dp.PierceWiseLinearFunction import LinearFunction, LARGE_NUMBER
from dnl.PredictPlusOptimize import initialize_parameters, MEDIAN_LOSS, converge, CONV_CONST, get_regret_worker, \
    PredictPlusOptModel
import numpy as np
import multiprocessing as mp

from dnl.PredictPlustOptimizeUtils import compute_C_k, compute_F_k
from dnl.Solver import get_optimization_objective
from dnl.Utils import TransitionPoint, get_train_test_split, save_results_list


class PredictOptDP:
    def __init__(self, alphas=None, const=None, opt_params=None, loss=MEDIAN_LOSS,

                 is_parallel=True, run_time_limit=100000,
                 verbose=False, is_Val=True):
        """

        :param alphas: model parameters
        :param const:  model constant
        :param capacities: capacity of the optimization problem
        :param max_step_size_magnitude: sample space upper bound
        :param min_step_size_magnitude: sample space lower step size
        :param step_size_divider:
        :param sampling_method:
        """
        self.alphas = alphas
        self.const = const
        self.opt_params = opt_params
        self.is_val = is_Val

        self.is_parallel = is_parallel
        self.run_time_limit = run_time_limit
        self.training_obj_value = []
        self.test_regrets = []
        self.val_regrets = []
        self.epochs = []
        self.sub_epochs = []
        self.run_time = []
        self.test_MSE = 0

        self.loss = loss
        self.test_regret = 0
        self.training_MSE = 0

        self.test_run_time = 0

        self.verbose = verbose

    def init_params_lin_regression(self, X, Y):
        """
        initialize the model with linear regression
        :param train_set:
        :return:
        """
        params = initialize_parameters(X, Y)

        self.__setattr__('alphas', params.get('alphas'))
        self.__setattr__('const', params.get('const'))
        self.__setattr__('capacities', params.get('capacities'))

    def coordinate_descent(self, train_X, train_Y, train_weights, val_X, val_Y, val_weights, print_test=False,
                           test_X=None, test_Y=None,
                           test_weights=None, core_number=7):
        """
               Uses coordinate descent to optimize parameters
               :param train_X: test set features
               :param train_Y: test set output
               :param train_weights:
               :return: profit: average profit of the training set
               """
        is_break = False
        self_decided_features = list(range(len(self.alphas)))
        prev_profit = -10
        model_params = {'alphas': self.alphas,
                        'const': self.const}
        profit = np.median(get_optimization_objective(X=train_X, Y=train_Y, weights=train_weights,
                                                      opt_params=self.opt_params, model_params=model_params))
        test_regret = np.median(self.get_regret(test_X, test_Y, test_weights))
        val_regret = np.median(self.get_regret(val_X, val_Y, val_weights))
        self.test_regrets.append(test_regret)
        self.training_obj_value.append(profit)
        self.run_time.append(0)
        self.epochs.append(0)
        self.sub_epochs.append(0)
        self.val_regrets.append(val_regret)
        start_time = time.time()
        print("original objective value: " + str(profit))
        EPOCH = 0

        if self.is_parallel:
            mypool = mp.Pool(processes=min(8, core_number))
        else:
            mypool = None
        print("------------------------")
        train_X_og = train_X
        train_Y_og = train_Y
        train_weights_og = train_weights
        sub_epoch = 0
        while not converge(profit, prev_profit, CONV_CONST, flag=False):
            if self.verbose:
                print(converge(profit, prev_profit, CONV_CONST, flag=False))
            profit = np.median(get_optimization_objective(X=train_X, Y=train_Y, weights=train_weights,
                                                          opt_params=self.opt_params, model_params=model_params))
            prev_profit = profit
            print("-----------------------")
            random.seed(time.time())
            random.shuffle(self_decided_features)
            for k in self_decided_features:
                if self.verbose:
                    print('updating parameter: {}'.format(k))
                model_params = {'alphas': self.alphas,
                                'const': self.const}
                current_alpha = self.alphas[k, 0]
                if self.is_parallel:
                    debug_time = time.time()
                    map_func = partial(get_and_clean_transition_points, model_params=model_params,
                                       k=k,
                                       opt_params=self.opt_params,
                                       current_alpha=current_alpha)
                    iter = [[benchmark_X, benchmark_Y, benchmark_weights] for
                            benchmark_X, benchmark_Y, benchmark_weights in
                            zip(train_X, train_Y, train_weights)]

                    best_transition_points_set = mypool.starmap(map_func, iter)
                    best_transition_points_set = set().union(*best_transition_points_set)
                    if self.verbose:
                        print('transition points are identified. RUN TIME: {}'.format((time.time() - debug_time)))
                        debug_time = time.time()
                    benchmark_best_transition_point = find_the_best_transition_point_benchmarks(train_X, train_Y,
                                                                                                k=k,
                                                                                                model_params=model_params,
                                                                                                train_weights=train_weights,
                                                                                                opt_params=self.opt_params,
                                                                                                transition_point_list=list(
                                                                                                    best_transition_points_set),
                                                                                                prev_profit=profit,
                                                                                                pool=mypool)
                    if self.verbose:
                        print('found the best transition point  RUN TIME: {}'.format((time.time() - debug_time)))
                self.alphas[k, 0] = benchmark_best_transition_point.x
                profit = benchmark_best_transition_point.true_profit
                # record data for each parameter update if its full batch

                if self.is_val:
                    # print('val')
                    val_regret = np.median(self.get_regret(val_X, val_Y, val_weights, pool=mypool))
                    self.val_regrets.append(val_regret)
                test_run_time = time.time()
                if print_test:
                    # print('test')
                    test_regret = np.median(self.get_regret(test_X, test_Y, test_weights, pool=mypool))
                    self.test_regrets.append(test_regret)
                    train_regret = np.median(self.get_regret(train_X, train_Y, train_weights, pool=mypool))
                    self.training_obj_value.append(train_regret)
                    if self.verbose:
                        print('updating parameter', k, 'test regret', test_regret)
                        print("Updating Parameter: " + str(k) + " profit: " + str(profit))
                self.test_run_time = self.test_run_time + time.time() - test_run_time

                sub_epoch = sub_epoch + 1

                self.sub_epochs.append(sub_epoch)
                self.epochs.append(EPOCH)
                self.run_time.append((time.time() - start_time - self.test_run_time))
                print("EPOCH:", EPOCH, "sub epoch:", sub_epoch, "objective value:", profit, 'val regret',
                      self.val_regrets[-1], 'test regret', self.test_regrets[-1], 'run time: ', self.run_time[-1],
                      flush=True)

                if self.run_time[-1] > self.run_time_limit:
                    is_break = True
                    break
            if is_break:
                break

            EPOCH = EPOCH + 1

        self.number_of_epochs = EPOCH
        print("EPOCH:", EPOCH, "objective value:", profit, 'val regret', self.val_regrets[-1], 'test regret',
              self.test_regrets[-1])
        print('Training finished ')
        print("-----------------------")

        if self.is_parallel:
            mypool.close()
        return profit

    def get_regret(self, X, Y, weights=None, pool=None):
        model_params = {'alphas': self.alphas,
                        'const': self.const}
        if pool is None:
            # print('X shape', X[0].shape)
            #
            # print('y shape', len(Y))
            average_objective_value_with_predicted_items = get_optimization_objective(X=X, Y=Y, weights=weights,
                                                                                      opt_params=self.opt_params,
                                                                                      model_params=model_params
                                                                                      )
            optimal_average_objective_value = Solver.get_optimal_average_objective_value(X=X, Y=Y, weights=weights,
                                                                                         opt_params=self.opt_params,
                                                                                         )
            # print('regret predicted item value set',average_objective_value_with_predicted_items,'regret with real item value',optimal_average_objective_value)
            # print('pred obj', np.sum(average_objective_value_with_predicted_items))
            # print('true obj', np.sum(optimal_average_objective_value))
            # print(optimal_average_objective_value - average_objective_value_with_predicted_items)
            # print('true obj', np.sum(optimal_average_objective_value))
            regret = np.median(optimal_average_objective_value - average_objective_value_with_predicted_items)
            # print('regret', regret)
            # print(regret)

        else:
            map_func = partial(get_regret_worker, model_params=model_params, opt_params=self.opt_params)
            iter = zip(X, Y, weights)
            # [[x, y] for x, y in zip([4, 1, 0], [5, 1, 1])]
            objective_values = pool.starmap(map_func, iter)
            objective_values_predicted_items, optimal_objective_values = zip(*objective_values)
            # print('optimal_average_objective_value', objective_values_predicted_items)
            # print('average_objective_value_with_predicted_items', optimal_objective_values)
            regret = np.median(
                np.concatenate(optimal_objective_values) - np.concatenate(objective_values_predicted_items))
            # print('true obj',np.sum(np.concatenate(optimal_objective_values)))
        self.test_regret = regret
        return regret

    def print(self):
        first_line = ['Run Time Limit', 'Parallelism', 'Test MSE']
        second_line = [
            self.run_time_limit, self.is_parallel, self.test_MSE]
        third_line = ['epochs', 'sub epochs', 'run time', 'training objective', 'test regret', 'val regret']
        rest = np.array(
            [self.epochs, self.sub_epochs, self.run_time, self.training_obj_value, self.test_regrets,
             self.val_regrets]).T.tolist()
        print = []
        print.append(first_line)
        print.append(second_line)
        print.append(third_line)
        print.extend(rest)
        return print

    def get_file_name(self, file_type='.csv'):
        file_name = 'DP' + file_type
        return file_name


def get_and_clean_transition_points(benchmark_X, benchmark_Y, benchmark_weights, model_params, k,
                                    opt_params,
                                    current_alpha):
    """

    :param benchmark_X:
    :param benchmark_Y:
    :param benchmark_weights:
    :param model_params:
    :param k: parameter k
    :param opt_params: capacity for knapsack
    :param current_alpha:
    :return: a set transition points
    """
    # plf_time = time.time()
    transition_points = get_transition_points(
        model_params=model_params, k=k,
        train_X=benchmark_X,
        train_weights=benchmark_weights, opt_params=opt_params)
    # print('plf time: {}'.format((time.time()-plf_time)))
    # clean_time = time.time()
    cleaner_transition_points = clean_transition_points(
        transition_points=transition_points,
        benchmark_X=benchmark_X,
        benchmark_Y=benchmark_Y, weights=benchmark_weights,
        model_params=model_params, opt_params=opt_params,
        current_alpha=current_alpha, k=k)
    # print('clean_time: {}'.format((time.time() - clean_time)))
    return cleaner_transition_points


def get_transition_points(model_params, k,
                          train_X,
                          train_weights, opt_params):
    """
    gets transition points from a piecewise linear function
    :param model_params:
    :param k:
    :param train_X:
    :param train_weights:
    :param opt_params:
    :return:
    """
    plf = create_plf(model_params, opt_params, k, train_X, train_weights)
    transition_points = convert_plf_to_transition_points(plf)
    return transition_points


def create_plf(model_params, opt_params, k, train_X, train_weights):
    """
    create piecewise linear function by solving knapsack with DP.
    :param model_params:
    :param opt_params:
    :param k:
    :param train_X:
    :param train_weights:
    :return:
    """
    lin_funcs = create_lin_func_from_predictions(model_params, train_X, k=k)
    capacity = opt_params.get('capacity')
    some_time = time.time()
    plf = KnapsackSolverDP(capacity, train_weights, lin_funcs)
    # print('sometime: '.format((time.time() - some_time)))
    return plf


def convert_plf_to_transition_points(plf):
    """
    identify mid points of intervals and return them as transition points
    :param plf:
    :return:
    """
    plf_transition_points = plf.transition_points
    mid_points = []
    for index, point in enumerate(plf_transition_points):
        if point < LARGE_NUMBER:
            next_point = plf_transition_points[index + 1]
            if point == -LARGE_NUMBER:
                mid_points.append(TransitionPoint(x=next_point - 10))
            else:
                mid_points.append(TransitionPoint(x=(point + next_point) / 2))
    return mid_points


def create_lin_func_from_predictions(model_params, train_X, k):
    """
    parameterize coefficients as linear functions
    :param model_params:
    :param train_X:
    :param k:
    :return:
    """
    alphas = model_params.get('alphas')
    const = model_params.get('const')
    current_alpha = alphas[k, 0]
    constants = compute_C_k(X=train_X, alphas=alphas, const=const, k=k, isSampling=False)
    slopes = compute_F_k(X=train_X, alpha=current_alpha, C_k=constants, k=k)

    lin_funcs = []
    for slope, constant in zip(slopes.flatten(), constants.flatten()):
        lin_funcs.append(LinearFunction(slope=slope, constant=constant))
    return lin_funcs


def clean_transition_points(transition_points, benchmark_X, benchmark_Y, weights, opt_params, model_params,
                            current_alpha, k=0):
    """
    get rid of redundant transition points.
    :param transition_points:
    :param benchmark_X:
    :param benchmark_Y:
    :param weights:
    :param opt_params:
    :param model_params:
    :param current_alpha:
    :param k:
    :return:
    """
    alphas = model_params.get('alphas')
    const = model_params.get('const')
    cleaner_transition_points = set()
    base_profit = np.median(Solver.get_optimization_objective(X=[benchmark_X], Y=[benchmark_Y], weights=weights,
                                                              model_params=model_params, opt_params=opt_params))

    for transition_point in transition_points:
        tmp_alphas = alphas.copy()
        tmp_alphas[k, 0] = transition_point.x
        tmp_model_params = {'alphas': tmp_alphas,
                            'const': const}
        transition_point.true_profit = Solver.get_optimization_objective(X=[benchmark_X], Y=[benchmark_Y],
                                                                         weights=weights,
                                                                         model_params=tmp_model_params,
                                                                         opt_params=opt_params)
        if transition_point.true_profit > base_profit:
            cleaner_transition_points.add(transition_point.x)
    if not cleaner_transition_points:
        cleaner_transition_points.add(float(current_alpha))
    return cleaner_transition_points


def find_the_best_transition_point_benchmarks(train_X, train_Y, model_params, transition_point_list,
                                              opt_params,
                                              train_weights, prev_profit, k, pool=None):
    """
    compare all transition points for all benchmarks, return the optimal one
    :param train_X:
    :param train_Y:
    :param model_params:
    :param transition_point_list:
    :param opt_params:
    :param train_weights:
    :param prev_profit:
    :param k:
    :param pool:
    :return:
    """
    alphas = model_params.get('alphas')

    best_average_profit = prev_profit
    best_transition_point = TransitionPoint(alphas[k, 0], true_profit=prev_profit)

    if not (len(transition_point_list) == 1 and alphas[k, 0] == transition_point_list[0]):
        if pool is not None:
            map_func = partial(find_the_best_transition_point_benchmarks_worker, train_X=train_X, train_Y=train_Y,
                               train_weights=train_weights, model_params=model_params, opt_params=opt_params, k=k)
            results = pool.map(map_func, transition_point_list)
            results.append(best_transition_point)

            # print('x', [transition_point.x for transition_point in results], ' objective_value' ,
            # [transition_point.true_profit for transition_point in results])
            best_transition_point = max(results, key=attrgetter('true_profit'))


        else:
            for transition_point_x in transition_point_list:
                transition_point = find_the_best_transition_point_benchmarks_worker(transition_point_x, train_X=train_X,
                                                                                    train_Y=train_Y,
                                                                                    train_weights=train_weights,
                                                                                    model_params=model_params,
                                                                                    opt_params=opt_params, k=k)
                if transition_point.true_profit > best_average_profit:
                    best_average_profit = transition_point.true_profit
                    best_transition_point = copy.deepcopy(transition_point)

    return best_transition_point


def find_the_best_transition_point_benchmarks_worker(transition_point_x, train_X, train_Y, train_weights, model_params,
                                                     opt_params, k):
    alphas = model_params.get('alphas')
    alphas[k, 0] = transition_point_x
    model_params['alphas'] = alphas

    average_profit = np.median(get_optimization_objective(X=train_X, Y=train_Y,
                                                          weights=train_weights, opt_params=opt_params,
                                                          model_params=model_params))
    # print('k: ' + str(k) + ' transition_point: ' + str(transition_point_x) + ' profit: ' + str(average_profit))
    return TransitionPoint(transition_point_x, true_profit=average_profit)


def train_and_test_DP_example(opt_params=None,
                              generate_weight=True, unit_weight=True,
                              print_test=True, kfold=0, file_folder="", core_number=7, noise_level=0, dataset=None,
                              is_shuffle=False, file_name_prefix=None,verbose=False):
    if dataset is None:
        dataset = get_energy_data('energy_data.txt', generate_weight=generate_weight, unit_weight=unit_weight,
                                  kfold=kfold, noise_level=noise_level)

    train_set, test_set = get_train_test_split(dataset, random_seed=42, is_shuffle=is_shuffle)

    X_train = train_set.get('X')
    Y_train = train_set.get('Y')

    X_val = X_train[0:2880, :]
    Y_val = Y_train[0:2880]

    X_train = X_train[2880:]
    Y_train = Y_train[2880:, :]

    benchmarks_train_X = train_set.get('benchmarks_X')
    benchmarks_train_Y = train_set.get('benchmarks_Y')
    benchmarks_weights_train = train_set.get('benchmarks_weights')

    benchmarks_val_X = benchmarks_train_X[0:60]
    benchmarks_val_Y = benchmarks_train_Y[0:60]
    benchmarks_weights_val = benchmarks_weights_train[0:60]

    benchmarks_train_X = benchmarks_train_X[60:]
    benchmarks_train_Y = benchmarks_train_Y[60:]
    benchmarks_weights_train = benchmarks_weights_train[60:]

    # benchmark_number = 1
    train_X = benchmarks_train_X
    train_Y = benchmarks_train_Y
    train_weights = benchmarks_weights_train

    val_X = benchmarks_val_X
    val_Y = benchmarks_val_Y
    val_weights = benchmarks_weights_val

    test_X = test_set.get('benchmarks_X')
    test_Y = test_set.get('benchmarks_Y')
    test_weights = test_set.get('benchmarks_weights')
    #
    test_MSE_X = test_set.get('X')
    test_MSE_Y = test_set.get('Y')

    # Initialize parameters and get coefficients

    baseline_model = PredictPlusOptModel(opt_params=opt_params)

    # initialize models
    baseline_model.init_params_lin_regression(X_train, Y_train)

    print('baseline regret: {}'.format(baseline_model.get_regret(X=test_X, Y=test_Y, weights=test_weights)))

    #
    # divide_and_conquer_greedy_model = PredictPlusOptModel(
    #     sampling_method=Sampling_Methods.DIVIDE_AND_CONQUER_GREEDY,
    #     min_step_size_magnitude=-1,
    #     max_step_size_magnitude=0,
    #     step_size_divider=10, opt_params=opt_params,
    #     learning_rate=0.1, beta=0, mini_batch_size=32,
    #     run_time_limit=12000, epoch_limit=3, verbose=False)
    #
    # divide_and_conquer_greedy_model.init_params_lin_regression(X_train, Y_train)
    #
    # divide_greedy_profit = divide_and_conquer_greedy_model.coordinate_descent(train_X=train_X, train_Y=train_Y,
    #                                                                           train_weights=train_weights, val_X=val_X,
    #                                                                           val_Y=val_Y, val_weights=val_weights,
    #                                                                           test_X=test_X, test_Y=test_Y,
    #                                                                           test_weights=test_weights,
    #                                                                           print_test=print_test,
    #                                                                           core_number=7)

    dp_model = PredictOptDP(opt_params=opt_params, verbose=verbose)
    dp_model.init_params_lin_regression(X_train, Y_train)
    dp_model.coordinate_descent(train_X=train_X, train_Y=train_Y, train_weights=train_weights, test_X=test_X,
                                test_Y=test_Y, test_weights=test_weights, val_X=val_X, val_Y=val_Y,
                                val_weights=val_weights, print_test=print_test, core_number=core_number)
    file_name = file_name_prefix + dp_model.get_file_name()
    save_results_list(file_folder=file_folder, file_name=file_name, results=dp_model.print())


def test_knapsack_weighted_DP_example(capacities=[12],
                                      kfolds=[0],
                                      core_number=7, is_shuffle=False,
                                      n_iter=1, noise_level=0, file_folder_prefix=None,verbose=False):
    # dataset = np.load('Data.npz')
    dataset = None

    for capacity in capacities:
        if file_folder_prefix is None:
            file_folder = 'Tests/Knapsack/weighted/c' + str(capacity) + '/dp/'
        else:
            file_folder = file_folder_prefix
        for n in range(n_iter):
            for kfold in kfolds:
                opt_params = get_opt_params_knapsack_DP(capacity=capacity)
                file_name_prefix = 'knapsackdp-wc' + str(capacity) + "-k" + str(kfold) + '-'
                train_and_test_DP_example(dataset=dataset, kfold=kfold, file_folder=file_folder,
                                          file_name_prefix=file_name_prefix,
                                          opt_params=opt_params,
                                          generate_weight=True, unit_weight=False, core_number=core_number,
                                          is_shuffle=is_shuffle, noise_level=noise_level,verbose=verbose)


if __name__ == '__main__':
    test_knapsack_weighted_DP_example(capacities=[12],
                                      kfolds=[0],
                                      core_number=7, is_shuffle=False,
                                      n_iter=1, noise_level=0,verbose=True)
