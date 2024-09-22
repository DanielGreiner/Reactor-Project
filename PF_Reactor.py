# Code for the Plug flow reactor model, simple 1D

#imports

from Setup.Basic_Kinetics import *
from Setup.Reactor_Constants import *
from Setup.ODE_Solvers import *

# Working conditions
TC = 100                                # ^C
T = 273.15 + TC                         # K

RK4()
Euler()

rate_constants = k_arr(T)

print(rate_constants)