import matplotlib.pyplot as plt
import math as m
import numpy as np
from PFR.Setup.ODE_Solvers import *


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


fig, axs = plt.subplots(2, 2, figsize=(11, 7))
counter = 0
for i in range(2):
    for j in range(2):
        counter += 1
        plt.subplot(2, 2, counter)
        times = np.linspace(0, 10, 10 ** counter + 1)
        exact = np.vstack((times, easyfun2(times, 5))).T
        stepsize = times[1] - times[0]
        easyplot = euler(times, easyfun, 5)

        plt.plot(exact[:, 0], exact[:, 1], label='exact')
        plt.plot(easyplot[:, 0], easyplot[:, 1], label='euler')

        plt.title('stepsize = ' + str(stepsize))
        plt.legend(loc='best')
for ax in axs.flat:
    ax.set(xlabel='steps', ylabel='value')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()
