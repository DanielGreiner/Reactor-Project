import matplotlib.pyplot as plt
import math as m
import numpy as np

from PFR.Setup.Basic_Kinetics import k_arr
from PFR.Setup.ODE_Solvers import *
from PFR.Setup.Reaction_Functions import reac_fun_pl


def easyfun(x):
    y = 2*x
    return y


def easyfun2(time, x0):
    return x0*m.e**(time*2)


'''
for i in range(4):
    times = np.linspace(0, 10, 10**(i+1) + 1)
    exact = np.vstack((times, easyfun2(times, 5))).T
    stepsize = times[1] - times[0]
    easyplot = euler(times, easyfun, 5)
    plt.figure(i+1)
    plt.title('stepsize = ' + str(stepsize))
    plt.plot(exact[:, 0], exact[:, 1], label='exact')
    plt.plot(easyplot[:, 0], easyplot[:, 1], label='euler')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
'''

T = 373.15
IC = np.array([2, 2, 0, 0, 0])
k_array = k_arr(T)

fig, axs = plt.subplots(2, 2, figsize=(11, 7))
counter = 0
for i in range(2):
    for j in range(2):
        counter += 1
        plt.subplot(2, 2, counter)
        times = np.linspace(0, 10, 10 ** counter + 1)
        exact = np.vstack((times, easyfun2(times, 5))).T
        stepsize = times[1] - times[0]
        # eulerplots = euler_single(times, easyfun, 5)
        # rk4plots = rungekutta4_single(times, easyfun, 5)
        eulerplots = euler(times, reac_fun_pl, IC, k_array)
        rk4plots = rungekutta4(times, reac_fun_pl, IC, k_array)
        # plt.plot(exact[:, 0], exact[:, 1], label='exact', color='k')
        plt.plot(eulerplots[:, -1], eulerplots[:, 1], label='euler')
        plt.plot(rk4plots[:, -1], rk4plots[:, 1], label='runge kutta', linestyle='dotted')
        plt.title('stepsize = ' + str(stepsize))
        plt.legend(loc='best')
for ax in axs.flat:
    ax.set(xlabel='steps', ylabel='value')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()

# plots show RK4 being 2 magnitudes more efficient in calculations
