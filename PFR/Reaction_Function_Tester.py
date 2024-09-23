import matplotlib.pyplot as plt
import numpy as np

from PFR.Setup.Basic_Kinetics import k_arr
from PFR.Setup.ODE_Solvers import euler
from PFR.Setup.Reaction_Functions import reac_fun_pl

T = 373.15
IC = np.array([2, 2, 0, 0, 0])
k_array = k_arr(T)
times = np.linspace(0, 10, 10 ** 2 + 1)
eulerplots = euler(times, reac_fun_pl, IC, k_array)

lineObjects = plt.plot(eulerplots[:, -1], eulerplots[:, :-1])
plt.legend(lineObjects, ('A', 'B', 'C', 'D', 'E'))
plt.show()
