
import math

from dnl.LinearFunction import LinearFunction

from dnl.Solver import get_optimization_objective, get_optimization_objective_for_samples

from dnl.Utils import Interval, Point, create_transition_points_from_intervals, TransitionPoint

"""
This file contains the Sampler object and functions related to compute optimization probelem of benchmarks
"""

DIVIDE_AND_CONQUER = 'DIVIDE_AND_CONQUER'
DIVIDE_AND_CONQUER_MAX = 'DIVIDE_AND_CONQUER_MAX'
DIVIDE_AND_CONQUER_GREEDY = 'DIVIDE_AND_CONQUER_GREEDY'

EXHAUSTIVE = 'EXHAUSTIVE'
EXHAUSTIVE_MAX = 'EXHAUSTIVE_MAX'
SAMPLE_RANGE_MULTIPLIER = 3

MID_TRANSITION_POINT_SELECTION = 'MID_POINT'
EDGE_TRANSITION_POINT_SELECTION = 'EDGE'


class Sampler:
    """
    Sampler class: This class/object is used for deciding sampling space and finding transition points.
    """

    def __init__(self, max_step_size_magnitude=0, min_step_size_magnitude=-1, step_size_divider=10,
                 sampling_method="DIVIDE_AND_CONQUER", transition_point_selection=MID_TRANSITION_POINT_SELECTION,
                 opt_params=None):
        self.max_step_size_magnitude = max_step_size_magnitude
        self.min_step_size_magnitude = min_step_size_magnitude
        self.step_size_divider = step_size_divider
        self.sampling_method = sampling_method
        self.transition_point_selection = transition_point_selection
        self.opt_params = opt_params

    def divide_and_conquer_search(self, model_params, k, train_X, train_Y, train_weights):
        """
        This methods approaches the transition point search from a divide and conquer approach. We exploit the convex
        behaviour of the predicted profits with respect to parameter change of the optimization problem. The
        precision/step size starts of big, bounded by max_step_size_magnitude. And step size is reduced iteratively
        until it reaches the desired precision bounded by  min_step_size_magnitude. Each iteration a sample range is
        generated bounded by the intervals of the previous iterations and the step size.


        :param alphas: parameters of the model
        :param k: the index of the parameter that is optimized
        :param const: constant of the model
        :param train_X: test set features
        :param train_Y: test set profits
        :param train_weights: test set weights
        :param capacities: capacity of the problem

        :return: item_sets(list): list of item sets, transition_points: a list of transition points,
        predicted_profits(list): predicted profits of the sampling points
        profits(list): true profits of the sampling points
        sample_space(list): all sample points
        IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
        To get the final results you would look for list(m,:).
        """
        alphas = model_params.get('alphas')
        const = model_params.get('const')
        # Compute the number of iterations needed
        M = math.ceil(math.log(10 ** (self.max_step_size_magnitude - self.min_step_size_magnitude),
                               self.step_size_divider)) + 1

        alpha_k = alphas[k, 0] + 10e-7
        # Initialize first sample range parameters
        sample_range = abs(alpha_k * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
        step_size = abs(alpha_k) * (10 ** self.max_step_size_magnitude)

        interval_mid_point = alpha_k
        interval_start = interval_mid_point - sample_range / 2
        interval_end = interval_mid_point + sample_range / 2

        # Initialize lists
        sample_spaces = [[] for i in range(M)]
        predicted_profits = [[] for i in range(M)]
        profits = [[] for i in range(M)]
        transition_intervals = [[] for i in range(M)]
        intervals = [[] for i in range(M)]

        intervals[0] = [Interval(Point(interval_start, 0), Point(interval_end, 0))]

        for i in range(M):
            for interval in intervals[i]:

                start_index = interval.starting_point.x
                end_index = interval.ending_point.x
                sample_size = int(math.ceil((end_index - start_index) / step_size))
                # We need at least 3 points to extract a transition point
                if sample_size > 2:
                    sample_space = [start_index + step_size * j for j in range(sample_size)]
                    if round(end_index, 3) > round(sample_space[-1], 3):
                        sample_space.append(end_index)
                    sample_spaces[i].extend(sample_space)
                    # compute the profits of the sample points
                    benchmark_profit, benchmark_predicted_profit = get_optimization_objective_for_samples(
                        benchmark_X=train_X,
                        benchmark_Y=train_Y,
                        benchmark_weights=train_weights,
                        model_params=model_params,
                        opt_params=self.opt_params,
                        sample_space=sample_space, k=k,
                        )
                    # Find transition intervals from the predicted profits
                    transition_intervals[i].extend(
                        find_transition_intervals(sample_space, benchmark_predicted_profit, benchmark_profit))

                    if i < M - 1:
                        intervals[i + 1] = transition_intervals[i]

                    predicted_profits[i].extend(benchmark_predicted_profit)
                    profits[i].extend(benchmark_profit)

            step_size = step_size / self.step_size_divider

        transition_points = create_transition_points_from_intervals(transition_intervals,
                                                                    selection_method=self.transition_point_selection)
        return transition_points, transition_intervals, predicted_profits, profits, sample_spaces

    def divide_and_conquer_greedy_search(self, model_params, k, train_X, train_Y, train_weights):
        """
        This methods approaches the transition point search from a divide and conquer approach. Same as the vanilla divide and conquer method,
        but stops at the first transition point which improves the current profit


        :param alphas: parameters of the model
        :param k: the index of the parameter that is optimized
        :param const: constant of the model
        :param train_X: test set features
        :param train_Y: test set profits
        :param train_weights: test set weights
        :param capacities: capacity of the problem

        :return: item_sets(list): list of item sets, transition_points: a list of transition points,
        predicted_profits(list): predicted profits of the sampling points
        profits(list): true profits of the sampling points
        sample_space(list): all sample points
        IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
        To get the final results you would look for list(m,:).
        """
        alphas = model_params.get('alphas')
        const = model_params.get('const')
        # Compute the number of iterations needed
        M = math.ceil(math.log(10 ** (self.max_step_size_magnitude - self.min_step_size_magnitude),
                               self.step_size_divider)) + 1

        alpha_k = alphas[k, 0] + 10e-7
        profit_alpha_k = get_optimization_objective(X=[train_X], Y=[train_Y], weights=train_weights,
                                                    model_params=model_params,
                                                    opt_params=self.opt_params)
        # Initialize first sample range parameters
        sample_range = abs(alpha_k * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
        step_size = abs(alpha_k) * (10 ** self.max_step_size_magnitude)

        interval_mid_point = alpha_k
        interval_start = interval_mid_point - sample_range / 2
        interval_end = interval_mid_point + sample_range / 2

        # Initialize lists
        sample_spaces = [[] for i in range(M)]
        predicted_profits = [[] for i in range(M)]
        profits = [[] for i in range(M)]
        transition_intervals = [[] for i in range(M)]
        intervals = [[] for i in range(M)]

        intervals[0] = [Interval(Point(interval_start, 0), Point(interval_end, 0))]

        for i in range(M):
            for interval in intervals[i]:

                start_index = interval.starting_point.x
                end_index = interval.ending_point.x
                sample_size = int(math.ceil((end_index - start_index) / step_size))
                # We need at least 3 points to extract a transition point
                if sample_size > 2:
                    sample_space = [start_index + step_size * j for j in range(sample_size)]
                    if end_index > sample_space[-1]:
                        sample_space.append(end_index)
                    sample_spaces[i].extend(sample_space)
                    # compute the profits of the sample points
                    benchmark_profit, benchmark_predicted_profit = get_optimization_objective_for_samples(
                        benchmark_X=train_X,
                        benchmark_Y=train_Y,
                        benchmark_weights=train_weights,
                        model_params=model_params,
                        opt_params=self.opt_params,
                        sample_space=sample_space, k=k,
                        )
                    # Find transition intervals from the predicted profits

                    if max(benchmark_profit) > profit_alpha_k:
                        transition_points = [[TransitionPoint(sample_space[benchmark_profit.argmax()],
                                                              true_profit=max(benchmark_profit))]]
                        return transition_points, transition_intervals, predicted_profits, profits, sample_spaces

                    transition_intervals[i].extend(
                        find_transition_intervals(sample_space, benchmark_predicted_profit, benchmark_profit))

                    if i < M - 1:
                        intervals[i + 1] = transition_intervals[i]

                    predicted_profits[i].extend(benchmark_predicted_profit)
                    profits[i].extend(benchmark_profit)

            step_size = step_size / self.step_size_divider

        transition_points = [[TransitionPoint(alpha_k, true_profit=profit_alpha_k)]]
        return transition_points, transition_intervals, predicted_profits, profits, sample_spaces

    def exhaustive_search(self, model_params, k, train_X, train_Y, train_weights):
        """
              This methods approaches the transition point search from an exhaustive approach. The sample size is bounded
              by max_step_size_magnitude. Step size is determined by min_step_size_magnitude. Each sample point is used
              for profit calculations. And then these profits and sample points are used to extract transition points.


              :param alphas: parameters of the model
              :param k: the index of the parameter that is optimized
              :param const: constant of the model
              :param train_X: test set features
              :param train_Y: test set profits
              :param train_weights: test set weights
              :param capacities: capacity of the problem

              :return: item_sets(list): list of item sets, transition_points: a list of transition points,
              predicted_profits(list): predicted profits of the sampling points
              profits(list): true profits of the sampling points
              sample_space(list): all sample points
              IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
              To get the final results you would look for list(m,:).
              """
        alphas = model_params.get('alphas')
        alpha_k = alphas[k, 0] + 10e-7
        start_index = alpha_k

        sample_range = abs(alpha_k * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
        step_size = abs(alpha_k * (10 ** self.min_step_size_magnitude))
        sample_size = abs(int(sample_range / step_size))
        sample_space = [(start_index + (i * step_size) - sample_range / 2) for i in
                        range(sample_size)]

        profits, predicted_profits = get_optimization_objective_for_samples(benchmark_X=train_X,
                                                                                      benchmark_Y=train_Y,
                                                                                      benchmark_weights=train_weights,
                                                                                      model_params=model_params,
                                                                                      opt_params=self.opt_params,
                                                                                      sample_space=sample_space, k=k,
                                                                                      )

        transition_intervals = [find_transition_intervals(sample_space, predicted_profits, profits)]
        transition_points = create_transition_points_from_intervals(transition_intervals,
                                                                    selection_method=self.transition_point_selection)

        return transition_points, transition_intervals, predicted_profits, profits, sample_space

    def get_transition_points(self, model_params, train_X, train_Y, train_weights, k):
        """
        This is a wrapper function, calls the related functions depending of the model.

        DIVIDE AND CONQUER: USES DIVIDE AND CONQUER search for finding transition points, returns all transition points.
        DIVIDE AND CONQUER MAX: USES DIVIDE AND CONQUER search for finding transition points, returns the transition point with the best profit.
        EXHAUSTIVE:USES EXHAUSTIVE search for finding transition points.
        EXHAUSTIVE MAX: USES EXHAUSTIVE search for finding transition points, returns the transition point with the best profit.

        :param alphas: parameters of the model
        :param k: the index of the parameter that is optimized
        :param const: constant of the model
        :param train_X: test set features
        :param train_Y: test set profits
        :param train_weights: test set weights
        :param capacities: capacity of the problem

        :return:
        """
        if self.sampling_method == DIVIDE_AND_CONQUER:
            return self.divide_and_conquer_search(model_params=model_params, k=k, train_X=train_X, train_Y=train_Y,
                                                  train_weights=train_weights)

        if self.sampling_method == DIVIDE_AND_CONQUER_GREEDY:
            return self.divide_and_conquer_greedy_search(model_params=model_params, k=k, train_X=train_X,
                                                         train_Y=train_Y,
                                                         train_weights=train_weights)

        elif self.sampling_method == DIVIDE_AND_CONQUER_MAX:
            alphas = model_params.get('alphas')
            profit = get_optimization_objective(X=[train_X], Y=[train_Y], weights=train_weights,
                                                opt_params=self.opt_params, model_params=model_params)

            transition_points, transition_intervals, predicted_regrets, regrets, plot_x = self.divide_and_conquer_search(
                model_params=model_params, k=k, train_X=train_X, train_Y=train_Y,
                train_weights=train_weights)
            best_transition_point = [
                [find_best_transition_point(transition_points, TransitionPoint(x=alphas[k, 0], true_profit=profit))]]
            return best_transition_point, transition_intervals, predicted_regrets, regrets, plot_x

        elif self.sampling_method == EXHAUSTIVE:
            return self.exhaustive_search(model_params=model_params, k=k, train_X=train_X, train_Y=train_Y,
                                          train_weights=train_weights)

        elif self.sampling_method == EXHAUSTIVE_MAX:
            alphas = model_params.get('alphas')
            profit = get_optimization_objective(X=[train_X], Y=[train_Y], weights=train_weights,
                                                model_params=model_params, opt_params=self.opt_params)

            transition_points, transition_intervals, predicted_regrets, regrets, plot_x = self.exhaustive_search(

                model_params=model_params, k=k, train_X=train_X, train_Y=train_Y,
                train_weights=train_weights)
            best_transition_point = [
                [find_best_transition_point(transition_points, TransitionPoint(x=alphas[k, 0], true_profit=profit))]]
            return best_transition_point, transition_intervals, predicted_regrets, regrets, plot_x


def find_best_transition_point(transition_points, alpha):
    """
    find the transition point with the best profit.
    :param transition_points: a list of transition points
    :param alpha: parameters of the model
    :return: best_point: transition point with the best profit.
    """
    max_profit = alpha.true_profit
    best_point = alpha
    for transition_point in transition_points[len(transition_points) - 1]:
        if transition_point.true_profit > max_profit:
            best_point = transition_point

    return best_point


def find_transition_intervals(alpha_samples, predicted_profits, profits):
    """
    given alpha samples and related predicted profits, tries to find transition intervals. We do it by building linear
    function parameters for each interval, and comparing intervals.
    :param profits: profits of the sampled alphas
    :param alpha_samples: a set of alpha samples
    :param predicted_profits: predicted costs of the sampled alphas
    :return: transition_points(
    list): a list of transition points.
    """
    transition_intervals = []
    is_prev_point_transition = False
    i = 0
    while i < (len(alpha_samples) - 2):
        sample_point_1 = TransitionPoint((alpha_samples[i]), predicted_profits[i], profits[i])
        sample_point_2 = TransitionPoint((alpha_samples[i + 1]), predicted_profits[i + 1], profits[i + 1])
        sample_point_3 = TransitionPoint((alpha_samples[i + 2]), predicted_profits[i + 2], profits[i + 2])

        lin_func_1 = LinearFunction(sample_point_1, sample_point_2)
        lin_func_2 = LinearFunction(sample_point_2, sample_point_3)
        if not lin_func_1.is_same(lin_func_2):
            if not is_prev_point_transition:
                first_transition_interval = Interval(sample_point_1,
                                                     sample_point_2)
            second_transition_interval = Interval(sample_point_2,
                                                  sample_point_3)
            if not is_prev_point_transition:
                transition_intervals.extend([first_transition_interval, second_transition_interval])
            else:
                transition_intervals.extend([second_transition_interval])
            is_prev_point_transition = True
        else:
            is_prev_point_transition = False
        i = i + 1

    return transition_intervals
