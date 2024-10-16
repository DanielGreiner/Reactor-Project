import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from PFR.Setup.Basic_Kinetics import k_arr
from PFR.Setup.ODE_Solvers import euler
from PFR.Setup.Reaction_Functions import reac_fun_pl

T = 373.15
IC = np.array([2, 2, 0, 0, 0])
k_array = k_arr(T)
tmax = 10
times = np.linspace(0, tmax, 10 ** 2 + 1)
eulerplots = euler(times, reac_fun_pl, 0, IC, k_array)

# Solve the ODE using solve_ivp
sol = solve_ivp(reac_fun_pl,[0, tmax], IC, args=[k_array], t_eval=times)

# Plot the results
C_temp = sol.y
V_vals = sol.t
C_vals = np.transpose(C_temp)

plt.figure()
lines = plt.plot(V_vals, C_vals[:])
plt.legend(lines, ('A', 'B', 'C', 'D', 'E'))
plt.xlabel("Reactor Volume (L)")
plt.ylabel("Concentration (mol/L)")
plt.ylim(0,)
plt.xlim(0,)

plt.figure()
lineObjects = plt.plot(eulerplots[:, -1], eulerplots[:, :-1])
plt.legend(lineObjects, ('A', 'B', 'C', 'D', 'E'))
plt.ylim(0,)
plt.xlim(0,)
plt.xlabel("Reactor Volume (L)")
plt.ylabel("Concentration (mol/L)")

plt.figure()
plt.plot(eulerplots[:, -1], eulerplots[:, 0], label='A euler')
plt.plot(V_vals, C_vals[:, 0], label='A scipy')
plt.xlabel("Reactor Volume (L)")
plt.ylabel("Concentration (mol/L)")
plt.ylim(0,)
plt.xlim(0,)
plt.legend()
plt.show()
