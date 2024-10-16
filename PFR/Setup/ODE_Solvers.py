'''First order solvers'''
# imports
import numpy as np


# Runge Kutta 4


def rungekutta4(step_array, function, *args):
    stepsize = step_array[1] - step_array[0]
    argument_0 = args[0]
    arguments = args[1:]
    result = np.zeros((len(step_array), len(argument_0)))
    result[0, :] = argument_0[:]

    if len(arguments) == 0:
        for i in range(1, len(step_array)):
            # stage values of RK4
            k1 = function(result[i - 1, :])
            k2 = function(result[i - 1, :] + 0.5 * stepsize * k1)
            k3 = function(result[i - 1, :] + 0.5 * stepsize * k2)
            k4 = function(result[i - 1, :] + stepsize * k3)
            stage_values = k1 + 2 * k2 + 2 * k3 + k4
            result[i, :] = result[i - 1, :] + stepsize/6 * stage_values
    else:
        for i in range(1, len(step_array)):
            # stage values of RK4
            k1 = function(result[i - 1, :], *arguments)
            k2 = function(result[i - 1, :] + 0.5 * stepsize * k1, *arguments)
            k3 = function(result[i - 1, :] + 0.5 * stepsize * k2, *arguments)
            k4 = function(result[i - 1, :] + stepsize * k3, *arguments)
            stage_values = k1 + 2 * k2 + 2 * k3 + k4
            result[i, :] = result[i - 1, :] + stepsize/6 * stage_values

    # print(result)
    combined_result = np.concatenate((result, step_array.reshape(-1, 1)), axis=1)
    return combined_result


def rungekutta4_single(step_array, function, *args):
    stepsize = step_array[1] - step_array[0]
    result = np.zeros(len(step_array))
    argument_0 = args[0]
    arguments = args[1:]
    result[0] = argument_0
    if len(arguments) == 0:
        for i in range(1, len(step_array)):
            # stage values of RK4
            k1 = function(result[i - 1])
            k2 = function(result[i - 1] + 0.5 * stepsize * k1)
            k3 = function(result[i - 1] + 0.5 * stepsize * k2)
            k4 = function(result[i - 1] + stepsize * k3)
            stage_values = k1 + 2 * k2 + 2 * k3 + k4
            result[i] = result[i - 1] + stepsize/6 * stage_values
    else:
        for i in range(1, len(step_array)):
            # stage values of RK4
            k1 = function(result[i - 1], *arguments)
            k2 = function(result[i - 1] + 0.5 * stepsize * k1, *arguments)
            k3 = function(result[i - 1] + 0.5 * stepsize * k2, *arguments)
            k4 = function(result[i - 1] + stepsize * k3, *arguments)
            stage_values = k1 + 2 * k2 + 2 * k3 + k4
            result[i] = result[i - 1] + stepsize/6 * stage_values

    combined_result = np.append(result, step_array)

    return combined_result


# euler method


def euler(step_array, function, *args):
    # calculate stepsize to iterate by
    stepsize = step_array[1] - step_array[0]
    argument_0 = args[1]
    arguments = args[2:]
    result = np.zeros((len(step_array), len(argument_0)))
    result[0, :] = argument_0[:]
    # print('first line set:')
    # print(result)
    # print('stepsize : ' + str(stepsize))
    # iterate over the function given step array
    if len(arguments) == 0:
        for i in range(1, len(step_array)):
            result[i, :] = result[i - 1, :] + function(0, result[i - 1, :]) * stepsize
    else:
        for i in range(1, len(step_array)):
            temp = function(0, result[i - 1, :], *arguments) * stepsize
            result[i, :] = result[i - 1, :] + temp
    # print(result)
    combined_result = np.concatenate((result, step_array.reshape(-1, 1)), axis=1)

    return combined_result


def euler_single(step_array, function, *args):
    # calculate stepsize to iterate by
    stepsize = step_array[1] - step_array[0]
    result = np.zeros(len(step_array))
    argument_0 = args[1]
    arguments = args[2:]
    result[0] = argument_0

    # iterate over the function given step array
    if len(arguments) == 0:
        for i in range(1, len(step_array)):
            result[i] = result[i - 1] + function(0, result[i - 1]) * stepsize
    else:
        for i in range(1, len(step_array)):
            result[i] = result[i - 1] + function(0, result[i - 1], *arguments) * stepsize

    combined_result = np.append(result, step_array)
    return combined_result


'''Second order solvers'''
