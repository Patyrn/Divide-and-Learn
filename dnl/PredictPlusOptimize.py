import copy
import random
import time
from functools import partial
from sklearn.utils import shuffle
import numpy as np
from sklearn import linear_model

from dnl import Sampling_Methods, Solver

from dnl.PredictPlustOptimizeUtils import compute_C_k
from dnl.Solver import get_optimization_objective
from dnl.Utils import TransitionPoint, get_mini_batches
from operator import attrgetter
import multiprocessing as mp

CONV_CONST = 10E-6
MEDIAN_LOSS = 'MEDIAN'
MEAN_LOSS = 'MEAN LOSS'


class PredictPlusOptModel:
    def __init__(self, alphas=None, const=None, opt_params=None, loss=MEDIAN_LOSS, max_step_size_magnitude=1,
                 min_step_size_magnitude=-1,
                 step_size_divider=10, sampling_method=Sampling_Methods.DIVIDE_AND_CONQUER,
                 is_parallel=True, learning_rate=0.1, mini_batch_size=32, beta=0, epoch_limit=3, run_time_limit=100000,
                 verbose=False, is_Val = True):
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
        self.step_size_divider = step_size_divider

        self.is_parallel = is_parallel
        if mini_batch_size == -1:
            self.learning_rate = 1
        else:
            self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.epoch_limit = epoch_limit
        self.run_time_limit = run_time_limit
        self.training_obj_value = []
        self.test_regrets = []
        self.val_regrets = []
        self.epochs = []
        self.sub_epochs = []
        self.run_time = []
        self.max_step_size_magnitude = max_step_size_magnitude
        self.min_step_size_magnitude = min_step_size_magnitude
        self.sampling_method = sampling_method
        self.test_MSE = 0

        self.loss = loss
        self.number_of_epochs = 0
        self.test_regret = 0
        self.training_MSE = 0

        self.test_run_time = 0

        self.beta = beta
        self.verbose = verbose

    def init_params_lin_regression(self, X,Y):
        """
        initialize the model with linear regression
        :param train_set:
        :return:
        """
        params = initialize_parameters(X,Y)

        self.__setattr__('alphas', params.get('alphas'))
        self.__setattr__('const', params.get('const'))
        self.__setattr__('capacities', params.get('capacities'))

    def coordinate_descent(self, train_X, train_Y, train_weights, val_X, val_Y, val_weights, print_test=False, test_X=None, test_Y=None,
                           test_weights=None, core_number=7):
        """
        Uses coordinate descent to optimize parameters
        :param train_X: test set features
        :param train_Y: test set output
        :param train_weights:
        :return: profit: average profit of the training set
        """
        # self_decided_features = [4, 5, 6, 7]
        # self_decided_features = range(8)
        # self_decided_features = [4]
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

        direction = np.zeros(len(self_decided_features))
        momentum = np.zeros(len(self_decided_features))
        sampler = Sampling_Methods.Sampler(max_step_size_magnitude=self.max_step_size_magnitude,
                                           min_step_size_magnitude=self.min_step_size_magnitude,
                                           step_size_divider=self.step_size_divider,
                                           sampling_method=self.sampling_method,
                                           opt_params=self.opt_params)
        if self.is_parallel:
            mypool = mp.Pool(processes=min(8, core_number))
        else:
            mypool = None
        print("------------------------")
        train_X_og = train_X
        train_Y_og = train_Y
        train_weights_og = train_weights
        if self.mini_batch_size == -1:
            mini_batch_size = len(train_Y)
        else:
            mini_batch_size = self.mini_batch_size
        mini_batches_X, mini_batches_Y, mini_batches_weights = get_mini_batches(X=train_X, Y=train_Y,
                                                                                weights=train_weights,
                                                                                size=mini_batch_size)
        sub_epoch = 0
        while (EPOCH < self.epoch_limit) and self.run_time[-1] < self.run_time_limit and not converge(profit, prev_profit, CONV_CONST, self.mini_batch_size):
            mini_batches_X, mini_batches_Y, mini_batches_weights = shuffle(mini_batches_X, mini_batches_Y,
                                                                           mini_batches_weights)
            for mini_batch_X, mini_batch_Y, mini_batch_weights in zip(mini_batches_X, mini_batches_Y,
                                                                      mini_batches_weights):
                train_X = mini_batch_X
                train_Y = mini_batch_Y
                train_weights = mini_batch_weights
                profit = np.median(get_optimization_objective(X=train_X, Y=train_Y, weights=train_weights,
                                                              opt_params=self.opt_params, model_params=model_params))
                # cut for minibatch

                prev_profit = profit

                print("-----------------------")

                # use for raandom
                # for k in random.sample(range(len(self.alphas)), len(self.alphas) - 1):

                # for k in range(len(self.alphas)):
                random.seed(time.time())
                random.shuffle(self_decided_features)
                for k in self_decided_features:
                    model_params = {'alphas': self.alphas,
                                    'const': self.const}
                    current_alpha = self.alphas[k, 0]
                    best_transition_points_set = set()
                    if self.is_parallel:
                        map_func = partial(get_and_clean_transition_points, sampler=sampler, model_params=model_params,
                                           k=k,
                                           opt_params=self.opt_params,
                                           current_alpha=current_alpha)
                        iter = [[benchmark_X, benchmark_Y, benchmark_weights] for
                                benchmark_X, benchmark_Y, benchmark_weights in
                                zip(train_X, train_Y, train_weights)]

                        best_transition_points_set = mypool.starmap(map_func, iter)
                        best_transition_points_set = set().union(*best_transition_points_set)

                        benchmark_best_transition_point = find_the_best_transition_point_benchmarks(train_X, train_Y,
                                                                                                    k=k,
                                                                                                    model_params=model_params,
                                                                                                    train_weights=train_weights,
                                                                                                    opt_params=self.opt_params,
                                                                                                    transition_point_list=list(
                                                                                                        best_transition_points_set),
                                                                                                    prev_profit=profit,
                                                                                                    pool=mypool)

                    else:

                        for benchmark_X, benchmark_Y, benchmark_weights in zip(train_X, train_Y, train_weights):
                            best_transition_point, __, predicted_regrets, regrets, plot_x = sampler.get_transition_points(
                                model_params=model_params, k=k,
                                train_X=benchmark_X,
                                train_Y=benchmark_Y,
                                train_weights=benchmark_weights)
                            best_transition_point_set_benchmark = clean_transition_points(
                                transition_points=best_transition_point[-1],
                                benchmark_X=benchmark_X,
                                benchmark_Y=benchmark_Y, weights=benchmark_weights,
                                model_params=model_params, opt_params=self.opt_params,
                                current_alpha=current_alpha)
                            best_transition_points_set = best_transition_points_set.union(
                                best_transition_point_set_benchmark)

                        # To reduce training time move this process to the sampling method so we dont iterate through transition points list twice

                        benchmark_best_transition_point = find_the_best_transition_point_benchmarks(train_X, train_Y,
                                                                                                    k=k,
                                                                                                    model_params=model_params,
                                                                                                    train_weights=train_weights,
                                                                                                    opt_params=self.opt_params,
                                                                                                    transition_point_list=list(
                                                                                                        best_transition_points_set),
                                                                                                    prev_profit=profit)
                    gradient = benchmark_best_transition_point.x - self.alphas[k, 0]
                    # print((
                    #         benchmark_best_transition_point.x - self.alphas[k, 0]))
                    # print('dir', direction[k])
                    # if abs(gradient) > 0:
                    #     gradient = gradient / abs(gradient)

                    direction[k] = -self.beta * momentum[k] - (1 - self.beta) * gradient
                    # print(momentum, gradient, direction)
                    # print('mom: ', momentum[k], 'dir: ', direction[k])
                    self.alphas[k, 0] = self.alphas[k, 0] - direction[k] * self.learning_rate
                    momentum[k] = direction[k] * 1
                    profit = benchmark_best_transition_point.true_profit
                    #record data for each parameter update if its full batch
                    if self.mini_batch_size == -1:
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
                              self.val_regrets[-1], 'test regret', self.test_regrets[-1], flush=True)


                if not self.mini_batch_size == -1:
                    # Record data after each batch for mini batches
                    if self.is_val:
                        # print('val')
                        val_regret = np.median(self.get_regret(val_X,val_Y,val_weights,pool=mypool))
                        self.val_regrets.append(val_regret)
                    test_run_time = time.time()
                    if (print_test):
                        # print('test')
                        test_regret = np.median(self.get_regret(test_X, test_Y, test_weights,pool=mypool))
                        self.test_regrets.append(test_regret)
                        train_regret = np.median(self.get_regret(train_X, train_Y, train_weights,pool=mypool))
                        self.training_obj_value.append(train_regret)
                        if self.verbose:
                            print('updating parameter', k, 'test regret', test_regret)
                            print("Updating Parameter: " + str(k) + " profit: " + str(profit))
                    self.test_run_time = self.test_run_time + time.time() - test_run_time

                    sub_epoch = sub_epoch + 1

                    self.sub_epochs.append(sub_epoch)
                    self.epochs.append(EPOCH)
                    self.run_time.append((time.time() - start_time - self.test_run_time))
                    print("EPOCH:", EPOCH, "sub epoch:", sub_epoch, "objective value:", profit, 'val regret', self.val_regrets[-1],'test regret', self.test_regrets[-1])
                if self.run_time[-1] > self.run_time_limit:
                    is_break = True
                    break
            if is_break:
                break
            EPOCH = EPOCH + 1

        self.number_of_epochs = EPOCH
        print("EPOCH:", EPOCH, "objective value:", profit, 'val regret', self.val_regrets[-1], 'test regret', self.test_regrets[-1])
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
            iter =  zip(X, Y, weights)
            # [[x, y] for x, y in zip([4, 1, 0], [5, 1, 1])]
            objective_values = pool.starmap(map_func, iter)
            objective_values_predicted_items, optimal_objective_values = zip(*objective_values)
            # print('optimal_average_objective_value', objective_values_predicted_items)
            # print('average_objective_value_with_predicted_items', optimal_objective_values)
            print(np.mean(np.concatenate(optimal_objective_values)))
            regret = np.median(np.concatenate(optimal_objective_values) - np.concatenate(objective_values_predicted_items))
            # print('true obj',np.sum(np.concatenate(optimal_objective_values)))
        self.test_regret = regret
        return regret

    def get_MSE(self, X, Y):
        predicted_values = compute_C_k(X.T, self.alphas, self.const, isSampling=False)
        MSE = np.mean((Y - predicted_values) ** 2)
        self.test_MSE = MSE
        return MSE

    def print(self):
        first_line = ['Method', 'Max Step Size Order', 'Min Step Size Order', 'Run Time Limit', 'Epoch Limit',
                      'Mini Batch Size', 'Learning rate', 'Parallelism', 'Test MSE']
        second_line = [self.sampling_method, self.max_step_size_magnitude, self.min_step_size_magnitude,
                       self.run_time_limit, self.epoch_limit, self.mini_batch_size, self.learning_rate,
                       self.is_parallel, self.test_MSE]
        third_line = ['epochs', 'sub epochs', 'run time', 'training objective', 'test regret', 'val regret']
        rest = np.array(
            [self.epochs, self.sub_epochs, self.run_time, self.training_obj_value, self.test_regrets, self.val_regrets]).T.tolist()
        print = []
        print.append(first_line)
        print.append(second_line)
        print.append(third_line)
        print.extend(rest)
        return print

    def get_file_name(self, file_type='.csv'):
        file_name = str(self.sampling_method) + '-' + str(self.max_step_size_magnitude) + str(
            self.min_step_size_magnitude) + file_type
        return file_name


def get_and_clean_transition_points(benchmark_X, benchmark_Y, benchmark_weights, sampler, model_params, k, opt_params,
                                    current_alpha):
    best_transition_point, __, predicted_regrets, regrets, plot_x = sampler.get_transition_points(
        model_params=model_params, k=k,
        train_X=benchmark_X,
        train_Y=benchmark_Y,
        train_weights=benchmark_weights)
    best_transition_point_set_benchmark = clean_transition_points(
        transition_points=best_transition_point[-1],
        benchmark_X=benchmark_X,
        benchmark_Y=benchmark_Y, weights=benchmark_weights,
        model_params=model_params, opt_params=opt_params,
        current_alpha=current_alpha)
    return best_transition_point_set_benchmark


def find_the_best_transition_point_benchmarks(train_X, train_Y, model_params, transition_point_list,
                                              opt_params,
                                              train_weights, prev_profit, k, pool=None):
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


def get_regret_worker(X, Y, weights, model_params, opt_params ):
    # print('im in worker')
    # print('X shape', X.shape)
    # print('y shape', Y.shape)
    # print('weights shape', weights.shape)
    average_objective_value_with_predicted_items = get_optimization_objective(X=[X], Y=[Y], weights=[weights],
                                                                              opt_params=opt_params,
                                                                              model_params=model_params
                                                                              )
    # print('finished average_objective_value_with_predicted_items')
    optimal_average_objective_value = Solver.get_optimal_average_objective_value(X=[X], Y=[Y], weights=[weights],
                                                                                 opt_params=opt_params,
                                                                                 )
    # print('finished working')
    return average_objective_value_with_predicted_items, optimal_average_objective_value


def converge(profit, prev_profit, conv_const, flag):
    """
    A method to determine if the algorithm has reached the convergence point. Not used at the moment, but will be used in the full algorithm
    :param cost:
    :param prev_cost:
    :param conv_const: Convergence limit
    :return: is_converge : boolean
    """
    if flag > 0:
        return False
    else:
        print('prev profit', prev_profit, 'profit' , profit)
        print('ratio', abs((profit - prev_profit) / profit))
        if abs((profit - prev_profit) / profit) < conv_const:
            is_converged = True
        else:
            is_converged = False

    return is_converged


def initialize_parameters(X,Y):
    """
    initialize the parameters of the predict-opt model, AKA first stage
    :param train_set: dictionary containing X, and Y
    :return: params: dictionary, has initialized parameters of the model.
    """

    model = linear_model.Ridge().fit(X, Y)
    coef = model.coef_
    const = model.intercept_

    params = {'alphas': coef.T,
              'const': const}
    return params


def clean_transition_points(transition_points, benchmark_X, benchmark_Y, weights, opt_params, model_params,
                            current_alpha):
    cleaner_transition_points = set()
    base_profit = np.median(Solver.get_optimization_objective(X=[benchmark_X], Y=[benchmark_Y], weights=weights,
                                                              model_params=model_params, opt_params=opt_params))

    for transition_point in transition_points:
        if transition_point.true_profit > base_profit:
            cleaner_transition_points.add(transition_point.x)
    if not cleaner_transition_points:
        cleaner_transition_points.add(float(current_alpha))
    return cleaner_transition_points
