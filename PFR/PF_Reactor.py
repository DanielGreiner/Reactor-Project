# Code for the Plug flow reactor model, simple 1D
import math as m

#imports

from PFR.Setup.Basic_Kinetics import *
from PFR.Setup.Reactor_Constants import *
from PFR.Setup.ODE_Solvers import *

# Working conditions
TC = 100                                # ^C
T = 273.15 + TC                         # K

