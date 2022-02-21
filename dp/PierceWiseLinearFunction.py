# -*- coding: utf-8 -*-
"""

"""

import copy
import random

LARGE_NUMBER = 10000000


class LinearFunction():
    __name__ = "linear_function"

    def __init__(self, slope, constant):
        self.slope = slope
        self.constant = constant

    def evaluate(self, x):
        return self.slope * x + self.constant


def areFloatsEqual(a, b, eps=0.001):
    ratio_limit = 99.99
    if a==0 or b==0:
        return a==b
    else:
        ratio = min((a), (b)) * 100 / max((a),
                                                (b))

    if ratio > ratio_limit:
        return True
    else:
        return False


def withinBounds(x, lower_bound, upper_bound):
    return (lower_bound <= x and x <= upper_bound) or areFloatsEqual(x, lower_bound) or areFloatsEqual(x, upper_bound)


def areLinearFunctionsEqual(lin_func1, lin_func2):
    return areFloatsEqual(lin_func1.slope, lin_func2.slope) and areFloatsEqual(lin_func1.constant, lin_func2.constant)


class PiercewiseLinearFunction():
    __name__ = "piercewise_linear_function"

    # def __init__(self, slope, constant):
    #   self.transition_points = [-self.LARGE_NUMBER, self.LARGE_NUMBER]
    #  self.functions = [LinearFunction(slope, constant)]

    # assert(self.checkState())

    def __init__(self, transition_points=None, functions=None):
        if transition_points is None:
            transition_points = [-LARGE_NUMBER, LARGE_NUMBER]
        if functions is None:
            function = LinearFunction(slope=0,constant=-LARGE_NUMBER)
            functions = [function]
        assert (transition_points[0] == -LARGE_NUMBER)
        assert (transition_points[-1] == LARGE_NUMBER)

        # should merge intervals, i.e. if we have two intervals [a, b] and [b, c],
        # and on both we have the same linear function, the merge into one interval [a, c]
        self.transition_points = [transition_points[0]]
        self.functions = [functions[0]]
        i = 1;
        while i < len(transition_points) - 1:
            a = transition_points[i]

            previous_func = self.functions[-1]
            new_func = functions[i]  # getFunctionForPointLeft(a)

            # only if the functions are different, add them, otherwise nothing
            if areLinearFunctionsEqual(previous_func, new_func) == False:
                self.transition_points.append(a)
                self.functions.append(copy.deepcopy(new_func))

            i = i + 1

        if self.transition_points[-1] != LARGE_NUMBER:
            self.transition_points.append(LARGE_NUMBER)

            # due to numerical errors, it could that there are multiple identical points
        # in this case, examine the identical points, and take the better function
        i = 0
        while i < len(self.transition_points) - 1:
            if areFloatsEqual(self.transition_points[i], self.transition_points[i + 1]):
                del self.transition_points[i + 1]

                p = self.transition_points[i]
                if self.functions[i].evaluate(p - 10) > self.functions[i + 1].evaluate(p - 10):
                    del self.functions[i + 1]
                else:
                    del self.functions[i]

            else:
                i = i + 1

        assert (self.checkState())

    def evaluate(self, x):
        assert (self.checkState())

        func = self.getFunctionForPointLeft(x)
        return func.evaluate(x)

    def checkState(self):
        test0 = len(self.transition_points) > 0
        if test0 == False:
            print("Transition points is empty??")

            # check that there sizes of the arrays matches - there should be one more point than functions
        test1 = (len(self.transition_points) == (len(self.functions) + 1))
        if test1 == False:
            print("Size mismatch between points and functions")
            print(self.transition_points)
            for func in self.functions:
                print("{}, {}".format(func.slope, func.constant))

        # check that the transition points are sorted
        test2 = True
        for i in range(len(self.transition_points) - 1):
            test2 = test2 and (self.transition_points[i] < self.transition_points[i + 1])
            # print("{} < {}: {}".format(self.transition_points[i], self.transition_points[i+1], self.transition_points[i] < self.transition_points[i+1]))

        if test2 == False:
            print("Points are not sorted??")
            print(self.transition_points)

        test3 = self.transition_points[0] == -LARGE_NUMBER and self.transition_points[-1] == LARGE_NUMBER
        if test3 == False:
            print("End points are not correct??")

        return test0 and test1 and test2 and test3

    # returns the function that would be used to evaluate point x
    # in case x is a transition point, then two functions could be returned
    # we return the function from the left
    def getFunctionForPointLeft(self, x):
        assert (self.checkState())
        return self.getFunctionForPointLeft_helper(self.transition_points, self.functions, x)

    def getFunctionForPointLeft_helper(self, points, funcs, x):
        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]

            if withinBounds(x=x, lower_bound=a, upper_bound=b) == True:
                return funcs[i]

        assert (1 == 2)  # 'x' should always be within one of the bounds - strange if it is not! ''

    def printToScreen(self):
        for i in range(len(self.transition_points) - 1):
            a = self.transition_points[i]
            b = self.transition_points[i + 1]
            func = self.functions[i]
            print("[{}, {}] -> s = {}, c = {}".format(a, b, func.slope, func.constant))
        print("-----")


def createBasicPiecewiseFunction(slope, constant):
    points = [-LARGE_NUMBER, LARGE_NUMBER]
    linFuncs = [LinearFunction(slope, constant)]
    return PiercewiseLinearFunction(points, linFuncs)


def getMergedTransitionPoints(plf1, plf2):
    merged_points = copy.deepcopy(plf1.transition_points)
    merged_points += plf2.transition_points
    merged_points = list(set(merged_points))
    merged_points.sort()
    return merged_points


def addLinearFunctions(func1, func2):
    slope_new = func1.slope + func2.slope
    constant_new = func1.constant + func2.constant
    return LinearFunction(slope=slope_new, constant=constant_new)


def addPiercewiseLinearFunctions(plf1, plf2):
    relevant_points = getMergedTransitionPoints(plf1, plf2)
    new_functions = []

    for i in range(1, len(relevant_points)):
        assert (relevant_points[i] > relevant_points[i - 1])
        a = relevant_points[i]

        lin_func1 = plf1.getFunctionForPointLeft(a)
        lin_func2 = plf2.getFunctionForPointLeft(a)

        new_func = addLinearFunctions(lin_func1, lin_func2)
        new_functions.append(new_func)

    return PiercewiseLinearFunction(transition_points=relevant_points, functions=new_functions)


def getBetterFunction(lin_func1, lin_func2, x):
    if lin_func1.evaluate(x) >= lin_func2.evaluate(x):
        return lin_func1
    else:
        return lin_func2


def getWorseFunction(lin_func1, lin_func2, x):
    if lin_func1.evaluate(x) < lin_func2.evaluate(x):
        return lin_func1
    else:
        return lin_func2


def computeIntersectionOfLinearFunctions(lin_func1, lin_func2):
    # if the lines are parallel, no intersection
    if areFloatsEqual(lin_func1.slope, lin_func2.slope):
        return None

    # p = (c2 - c1)/(a1 - a2)
    intersection = float(lin_func2.constant - lin_func1.constant) / (lin_func1.slope - lin_func2.slope)

    # just verify that this is indeed an intersection point
    if areFloatsEqual(lin_func1.evaluate(intersection), lin_func2.evaluate(intersection)) == False:
        print(lin_func1.evaluate(intersection))
        print(lin_func2.evaluate(intersection))
        assert (areFloatsEqual(lin_func1.evaluate(intersection), lin_func2.evaluate(intersection)))

    return intersection


def checkSorted(array, k1, k2):
    for i in range(len(array) - 1):
        if array[i] > array[i + 1]:
            print(k1)
            print(k2)
            print(array[i])
            print(array[i + 1])
            assert (1 == 2)


def maxPiercewiseLinearFunctions(plf1, plf2):
    relevant_points = getMergedTransitionPoints(plf1, plf2)
    new_points = [-LARGE_NUMBER]
    new_functions = []

    # print("relevant points: {}".format(relevant_points))

    for i in range(1, len(relevant_points)):
        assert (relevant_points[i - 1] < relevant_points[i])

        a = relevant_points[i - 1]
        b = relevant_points[i]

        lin_func1 = plf1.getFunctionForPointLeft(b)
        lin_func2 = plf2.getFunctionForPointLeft(b)

        # intersection point will be None if the two functions are parallel, i.e. there is no intersection
        intersection_point = computeIntersectionOfLinearFunctions(lin_func1, lin_func2)
        # print("inter {}".format(intersection_point))

        # clamp the intersection if it is similar
        if intersection_point != None and areFloatsEqual(intersection_point, a):
            intersection_point = a

        if intersection_point != None and areFloatsEqual(intersection_point, b):
            intersection_point = b

        if intersection_point == None or withinBounds(x=intersection_point, lower_bound=a, upper_bound=b) == False:
            f_best = getBetterFunction(lin_func1, lin_func2, (
                    a + b) / 2)  # evaluate the better function on a point in the interval -> they the functions intersect outside of the interval, only one will be best on the interval
            new_points.append(b)
            new_functions.append(f_best)
        else:

            # corner case - if the two points are equal, then only look at the left side
            if areFloatsEqual(intersection_point, b):
                new_points.append(b)
                f_left = getBetterFunction(lin_func1, lin_func2,
                                           intersection_point - 10)  # the minus ten is just to indicate from the left side, can be small epsilon
                new_functions.append(f_left)
            elif areFloatsEqual(intersection_point, a):  # look only to the right
                new_points.append(b)
                f_right = getBetterFunction(lin_func1, lin_func2, intersection_point + 10)
                new_functions.append(f_right)
            else:
                assert (intersection_point < b)
                new_points.append(intersection_point)
                new_points.append(b)

                f_left = getBetterFunction(lin_func1, lin_func2,
                                           intersection_point - 10)  # the minus ten is just to indicate from the left side, can be small epsilon
                f_right = getBetterFunction(lin_func1, lin_func2, intersection_point + 10)

                new_functions.append(f_left)
                new_functions.append(f_right)

            # checkSorted(new_points, intersection_point, b)

    assert (new_points[-1] == LARGE_NUMBER)
    return PiercewiseLinearFunction(transition_points=new_points, functions=new_functions)


def minPiercewiseLinearFunctions(plf1, plf2):
    relevant_points = getMergedTransitionPoints(plf1, plf2)
    new_points = [-LARGE_NUMBER]
    new_functions = []

    # print("relevant points: {}".format(relevant_points))

    for i in range(1, len(relevant_points)):
        assert (relevant_points[i - 1] < relevant_points[i])

        a = relevant_points[i - 1]
        b = relevant_points[i]

        lin_func1 = plf1.getFunctionForPointLeft(b)
        lin_func2 = plf2.getFunctionForPointLeft(b)

        # intersection point will be None if the two functions are parallel, i.e. there is no intersection
        intersection_point = computeIntersectionOfLinearFunctions(lin_func1, lin_func2)
        # print("inter {}".format(intersection_point))

        # clamp the intersection if it is similar
        if intersection_point != None and areFloatsEqual(intersection_point, a):
            intersection_point = a

        if intersection_point != None and areFloatsEqual(intersection_point, b):
            intersection_point = b

        if intersection_point == None or withinBounds(x=intersection_point, lower_bound=a, upper_bound=b) == False:
            f_min = getWorseFunction(lin_func1, lin_func2, (
                    a + b) / 2)  # evaluate the better function on a point in the interval -> they the functions intersect outside of the interval, only one will be best on the interval
            new_points.append(b)
            new_functions.append(f_min)
        else:

            # corner case - if the two points are equal, then only look at the left side
            if areFloatsEqual(intersection_point, b):
                new_points.append(b)
                f_left = getWorseFunction(lin_func1, lin_func2,
                                          intersection_point - 10)  # the minus ten is just to indicate from the left side, can be small epsilon
                new_functions.append(f_left)
            elif areFloatsEqual(intersection_point, a):  # look only to the right
                new_points.append(b)
                f_right = getWorseFunction(lin_func1, lin_func2, intersection_point + 10)
                new_functions.append(f_right)
            else:
                assert (intersection_point < b)
                new_points.append(intersection_point)
                new_points.append(b)

                f_left = getWorseFunction(lin_func1, lin_func2,
                                          intersection_point - 10)  # the minus ten is just to indicate from the left side, can be small epsilon
                f_right = getWorseFunction(lin_func1, lin_func2, intersection_point + 10)

                new_functions.append(f_left)
                new_functions.append(f_right)

            # checkSorted(new_points, intersection_point, b)

    assert (new_points[-1] == LARGE_NUMBER)
    return PiercewiseLinearFunction(transition_points=new_points, functions=new_functions)


def generateRandomLinearFunction():
    slope = random.randint(1, 20)
    constant = random.randint(1, 30)
    return LinearFunction(slope, constant)


def convertLinFunctionToPLF(linear_function):
    transition_points = [-LARGE_NUMBER, LARGE_NUMBER]
    functions = [copy.deepcopy(linear_function)]
    return PiercewiseLinearFunction(transition_points, functions)


def createLargeConstantFunction():
    return convertLinFunctionToPLF(LinearFunction(slope=0, constant=LARGE_NUMBER))


def createZeroLinearFunction():
    lin_func = LinearFunction(0, 0)
    return convertLinFunctionToPLF(lin_func)


def doBasicTests():
    lin_funcs = [generateRandomLinearFunction() for i in range(8)]
    plfs = [convertLinFunctionToPLF(lin_funcs[i]) for i in range(8)]

    plf1 = maxPiercewiseLinearFunctions(plfs[0], plfs[1])
    plf2 = maxPiercewiseLinearFunctions(plfs[2], plfs[3])

    plf3 = addPiercewiseLinearFunctions(plf1, plf2)

    plf4 = maxPiercewiseLinearFunctions(plfs[4], plfs[5])
    plf5 = maxPiercewiseLinearFunctions(plfs[6], plfs[7])

    plf6 = addPiercewiseLinearFunctions(plf4, plf5)

    plf7 = maxPiercewiseLinearFunctions(plf6, plf3)

    step_size = 1
    x = -LARGE_NUMBER
    while x < LARGE_NUMBER:
        print(x)
        assert (plf3.evaluate(x) == plf2.evaluate(x) + plf1.evaluate(x))
        assert (plf1.evaluate(x) == max([lin_funcs[0].evaluate(x), lin_funcs[1].evaluate(x)]))
        assert (plf2.evaluate(x) == max([lin_funcs[2].evaluate(x), lin_funcs[3].evaluate(x)]))

        assert (plf6.evaluate(x) == plf4.evaluate(x) + plf5.evaluate(x))
        assert (plf4.evaluate(x) == max([lin_funcs[4].evaluate(x), lin_funcs[5].evaluate(x)]))
        assert (plf5.evaluate(x) == max([lin_funcs[6].evaluate(x), lin_funcs[7].evaluate(x)]))

        if (plf7.evaluate(x) != max([plf3.evaluate(x), plf6.evaluate(x)])):
            # print("Point {}".format(x))

            # print(plf3.evaluate(x))
            # print(plf6.evaluate(x))
            # print(plf7.evaluate(x))

            # print(plf3.transition_points)
            # print(plf6.transition_points)
            # print(plf7.transition_points)
            plf3.printToScreen()
            plf6.printToScreen()
            plf7.printToScreen()

            assert (plf7.evaluate(x) == max([plf3.evaluate(x), plf6.evaluate(x)]))

        x += step_size


def exampleFunc(alpha):
    x1 = -1 * alpha + 10
    x2 = alpha + 2
    x3 = -0.5 * alpha + 5
    x4 = 2 * alpha - 5

    print("{} -> {}".format(alpha, [x1, x2, x3, x4]))


if __name__ == '__main__':
    exampleFunc(2)


