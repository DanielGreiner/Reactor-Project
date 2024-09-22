import matplotlib.pyplot as plt
import math as m
import numpy as np
from PFR.Setup.ODE_Solvers import *


def easyfun(x):
    y = 2*x
    return y


def easyfun2(time, x0):
    return x0*m.e**(time*2)


for i in range(4):
    times = np.linspace(0, 10, 10**(i+1) + 1)
    exact = np.vstack((times, easyfun2(times, 5))).T
    stepsize = times[1] - times[0]
    easyplot = euler(times, easyfun, 5)
    plt.figure(i)
    plt.title('stepsize = ' + str(stepsize))
    plt.plot(easyplot[:, 0], easyplot[:, 1], label='euler')
    plt.plot(exact[:, 0], exact[:, 1], label='exact')
    plt.legend()
plt.show()


