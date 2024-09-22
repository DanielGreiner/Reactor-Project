#### First order solvers
# imports
import numpy as np


# Runge Kutta 4


def rungekutta4():
    print('RK4')

# Euler


def euler(step_array, function, *args):
    # calculate stepsize to iterate by
    stepsize = step_array[1] - step_array[0]
    result = np.zeros(len(step_array))
    argument_0 = args[0]
    arguments = args[1:]
    result[0] = argument_0

    # iterate over the function given step array
    if len(arguments) == 0:
        for i in range(1, len(step_array)):
            result[i] = result[i - 1] + function(result[i - 1]) * stepsize
    else:
        for i in range(1, len(step_array)):
            result[i] = result[i - 1] + function(result[i - 1], *arguments) * stepsize

    combined_result = np.vstack((step_array, result)).T
    #print('Euler: ', combined_result)
    return combined_result


#### Second order solvers

