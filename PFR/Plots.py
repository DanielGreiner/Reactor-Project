from PF_Reactor import *

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def pftr_model(V, C, k, F_A0):
    """
    ODE function for the PFTR model.

    Parameters:
    V : float
        Reactor volume (independent variable).
    C : list
        Concentration of species (dependent variable), [C_A].
    k : float
        Reaction rate constant.
    F_A0 : float
        Initial molar flow rate of species A.

    Returns:
    dC_A_dV : float
        Rate of change of concentration of A with respect to reactor volume.
    """
    C_A = C[0]  # Concentration of A
    dC_A_dV = -k * C_A / F_A0
    return [dC_A_dV]


def solve_pftr(k, F_A0, C_A0, V_max, num_points=100):
    """
    Solves the PFTR model ODE and plots the concentration profile with specified number of points.

    Parameters:
    k : float
        Reaction rate constant (1/s).
    F_A0 : float
        Initial molar flow rate of A (mol/s).
    C_A0 : float
        Initial concentration of A (mol/L).
    V_max : float
        Total reactor volume (L).
    num_points : int, optional
        Number of points to evaluate along the reactor volume (default is 100).

    Returns:
    result : OdeResult
        The result of the ODE solver, including the volumes and concentrations.
    """
    # Initial conditions
    C0 = [C_A0]  # Initial concentration of A

    # Create a volume grid where we want the solution
    V_grid = np.linspace(0, V_max, num_points)

    # Solve the ODE using solve_ivp
    sol = solve_ivp(
        fun=pftr_model,  # ODE function
        t_span=[0, V_max],  # Volume range (0 to V_max)
        y0=C0,  # Initial condition
        args=(k, F_A0),  # Additional arguments to the ODE function
        t_eval=V_grid,  # Specify volume points to evaluate
        method='RK45',  # Runge-Kutta solver
        dense_output=True  # Generate a continuous solution
    )

    # Plot the results
    V_vals = sol.t
    C_A_vals = sol.y[0]

    plt.plot(V_vals, C_A_vals, label="Concentration of A")
    plt.xlabel("Reactor Volume (L)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile in a PFTR")
    plt.legend()
    plt.grid()
    plt.show()

    return sol


# Example usage:
k = 0.3  # Rate constant (1/s)
F_A0 = 1.0  # Molar flow rate of A (mol/s)
C_A0 = 2.0  # Initial concentration of A (mol/L)
V_max = 10.0  # Total reactor volume (L)

solve_pftr(k, F_A0, C_A0, V_max)