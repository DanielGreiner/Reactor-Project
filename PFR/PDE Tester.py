import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# steady state


def pftr_dispersion_model(z, y, u, D, k):
    """
    ODE system for the PFTR model with axial dispersion.

    Parameters:
    z : float
        Axial position in the reactor (m).
    y : list
        Concentration and its first derivative, [C_A, dC_A/dz].
    u : float
        Fluid velocity (m/s).
    D : float
        Axial dispersion coefficient (m^2/s).
    k : float
        Reaction rate constant (1/s).

    Returns:
    dydz : list
        List containing first derivative of concentration and second derivative of concentration.
    """
    C_A, dC_A_dz = y  # y[0] = C_A, y[1] = dC_A/dz

    # Define the system of equations
    dC_A_dz_new = dC_A_dz
    ddC_A_dz2_new = (u / D) * dC_A_dz + (k / D) * C_A

    return [dC_A_dz_new, ddC_A_dz2_new]


def solve_pftr_dispersion(u, D, k, C_A0, dC_A_dz0, z_max, num_points=100):
    """
    Solves the PFTR model ODE with axial dispersion and plots the concentration profile.

    Parameters:
    u : float
        Fluid velocity (m/s).
    D : float
        Axial dispersion coefficient (m^2/s).
    k : float
        Reaction rate constant (1/s).
    C_A0 : float
        Initial concentration of A at z=0 (mol/L).
    dC_A_dz0 : float
        Initial concentration gradient at z=0 (mol/L/m).
    z_max : float
        Maximum axial length of the reactor (m).
    num_points : int, optional
        Number of points to evaluate along the reactor length (default is 100).

    Returns:
    result : OdeResult
        The result of the ODE solver, including the axial positions and concentrations.
    """
    # Initial conditions
    y0 = [C_A0, dC_A_dz0]  # [C_A(z=0), dC_A/dz(z=0)]

    # Create a grid for z (axial position)
    z_grid = np.linspace(0, z_max, num_points)

    # Solve the ODE system using solve_ivp
    sol = solve_ivp(
        fun=pftr_dispersion_model,  # ODE function
        t_span=[0, z_max],  # Axial length range (0 to z_max)
        y0=y0,  # Initial condition
        args=(u, D, k),  # Additional arguments to the ODE function
        t_eval=z_grid,  # Specify z points to evaluate
        method='RK45'  # Runge-Kutta solver
    )

    # Extract results
    z_vals = sol.t
    C_A_vals = sol.y[0]  # Concentration of A

    # Plot the results
    plt.plot(z_vals, C_A_vals, label="Concentration of A with Axial Dispersion")
    plt.xlabel("Axial Position (m)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile in a PFTR with Axial Dispersion")
    plt.legend()
    plt.grid()
    plt.show()

    return sol


# Example usage:
u = 0.1  # Fluid velocity (m/s)
D = 0.01  # Axial dispersion coefficient (m^2/s)
k = 0.1  # Reaction rate constant (1/s)
C_A0 = 2.0  # Initial concentration of A at z=0 (mol/L)
dC_A_dz0 = 0.0  # Initial concentration gradient at z=0 (mol/L/m)
z_max = 10.0  # Total reactor length (m)

solve_pftr_dispersion(u, D, k, C_A0, dC_A_dz0, z_max)


# non-steady state

def pftr_pde_system(t, C_A, u, D, k, N, dz):
    """
    Defines the system of ODEs derived from the PDE for non-steady-state PFTR with axial dispersion.

    Parameters:
    t : float
        Time (s).
    C_A : array
        Concentrations of A at all spatial points along the reactor length.
    u : float
        Fluid velocity (m/s).
    D : float
        Axial dispersion coefficient (m^2/s).
    k : float
        Reaction rate constant (1/s).
    N : int
        Number of spatial grid points.
    dz : float
        Spatial step size (m).

    Returns:
    dC_A_dt : array
        Time derivatives of the concentrations at each spatial point.
    """
    dC_A_dt = np.zeros(N)

    # Apply finite difference method for internal points (central difference for second derivative)
    for i in range(1, N - 1):
        dC_A_dt[i] = (
                D * (C_A[i + 1] - 2 * C_A[i] + C_A[i - 1]) / dz ** 2  # Diffusion term
                - u * (C_A[i] - C_A[i - 1]) / dz  # Convection term
                - k * C_A[i]  # Reaction term
        )

    # Boundary conditions
    dC_A_dt[0] = 0  # No-flux boundary condition at the inlet (can be changed based on the system)
    dC_A_dt[-1] = 0  # No-flux boundary condition at the outlet (can be changed based on the system)

    return dC_A_dt


def solve_nonsteady_pftr(u, D, k, C_A0, z_max, t_max, dz, num_time_points=100):
    """
    Solves the non-steady-state PFTR model ODE system with axial dispersion and plots the concentration profile over time.

    Parameters:
    u : float
        Fluid velocity (m/s).
    D : float
        Axial dispersion coefficient (m^2/s).
    k : float
        Reaction rate constant (1/s).
    C_A0 : float
        Initial concentration of A along the reactor (mol/L).
    z_max : float
        Maximum reactor length (m).
    t_max : float
        Total simulation time (s).
    dz : float
        Spatial step size (m).
    num_time_points : int, optional
        Number of time points to evaluate (default is 100).

    Returns:
    sol : OdeResult
        The result of the ODE solver.
    z_grid : array
        Spatial grid points.
    """
    # Create spatial grid
    z_grid = np.arange(0, z_max + dz, dz)
    N = len(z_grid)  # Number of spatial points

    # Initial condition: Concentration of A at time t=0
    C_A_init = np.full(N, C_A0)

    # Time grid
    t_grid = np.linspace(0, t_max, num_time_points)

    # Solve the system of ODEs using solve_ivp
    sol = solve_ivp(
        fun=pftr_pde_system,  # ODE function
        t_span=[0, t_max],  # Time range
        y0=C_A_init,  # Initial condition
        args=(u, D, k, N, dz),  # Additional arguments to the ODE function
        t_eval=t_grid,  # Time points to evaluate
        method='RK45'  # Runge-Kutta solver
    )

    # Plot the results
    for i, t in enumerate(sol.t):
        if i % (len(sol.t) // 10) == 0:  # Plot every 10th time point
            plt.plot(z_grid, sol.y[:, i], label=f't = {t:.2f} s')

    plt.xlabel("Axial Position (m)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile in a PFTR with Axial Dispersion (Non-Steady-State)")
    plt.legend()
    plt.grid()
    plt.show()

    return sol, z_grid


# Example usage:
u = 0.1  # Fluid velocity (m/s)
D = 0.01  # Axial dispersion coefficient (m^2/s)
k = 0.1  # Reaction rate constant (1/s)
C_A0 = 2.0  # Initial concentration of A (mol/L) along the reactor
z_max = 10.0  # Total reactor length (m)
t_max = 50.0  # Total simulation time (s)
dz = 0.1  # Spatial step size (m)

solve_nonsteady_pftr(u, D, k, C_A0, z_max, t_max, dz)
