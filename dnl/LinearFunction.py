class LinearFunction:

    def __init__(self, point_1=None, point_2=None):
        if point_1 is not None and point_2 is not None:
            (self.slope, self.constant) = get_lin_func_param(point_1, point_2)
        else:
            self.slope = 0
            self.constant = 0

    def set_lin_func_param_from_points(self, point_1, point_2):
        """
        given two pairs of (X,Y) extracts parameters of a linear function.
        m: slope
        b: constant
        :param point_2:
        :param point_1:
        :return: m, b
        """
        (slope, constant) = self.get_lin_func_param(point_1, point_2)

        self.slope = slope
        self.constant = constant

    def set_lin_func_param(self, slope, constant):
        """
        given two pairs of (X,Y) extracts parameters of a linear function.
        m: slope
        b: constant
        :param point_2:
        :param point_1:
        :return: m, b
        """
        self.slope = slope
        self.constant = constant

    def is_same(self, lin_func_2):
        """
        given parameters of two linear function, evaluates whether they are same.
        Right not there are some division errors and exact comparison fails to prove to be robust.
        Not checking constants for now, it is not likely to have same line. Otherwise floating number arithmetics cause problems.
        :param lin_func_2:
        :param lin_func_1:

        :return:
        """
        return is_same_slope(self.slope, lin_func_2.slope, self.constant, lin_func_2.constant)

def initialize_lin_func(point_1, point_2, point_3):
    """
    When first initializing a linear function we need to compare three points. (Actually may not be true check)
    :param X1:
    :param X2:
    :param X3:
    :param Y1:
    :param Y2:
    :param Y3:
    :return:
    """
    slope_1, constant_1 = get_lin_func_param(point_1, point_2)
    slope_2, constant_2 = get_lin_func_param(point_2, point_3)
    return is_same_slope(slope_1, slope_2, constant_1, constant_2), slope_1, constant_1




def is_same_slope(slope_1, slope_2, constant_1, constant_2):
    """
    given parameters of two linear function, evaluates whether they are same.
    Right not there are some division errors and exact comparison fails to prove to be robust.
    Not checking constants for now, it is not likely to have same line. Otherwise floating number arithmetics cause problems.
    :param lin_func_2:
    :param lin_func_1:

    :return:
    """
    ratio_limit = 99.99
    if slope_1 == 0 or slope_2 == 0:
        if slope_1 == slope_2 and constant_1 == constant_2:
            return True
        else:
            return False

    ratio = min((slope_1), (slope_2)) * 100 / max((slope_1),
                                                        (slope_2))
    if ratio > ratio_limit:
        return True
    else:
        return False


def get_lin_func_param(point_1, point_2):
    """
    given two pairs of (X,Y) extracts parameters of a linear function.
    m: slope
    b: constant
    :param point_2:
    :param point_1:
    :return: slope, constant
    """

    precision = 5
    slope = round((point_1.predicted_profit - point_2.predicted_profit) / (point_1.x - point_2.x), precision)
    constant = round(point_1.predicted_profit - point_1.x * slope, precision)
    return slope, constant
