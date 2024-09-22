# Well hello there!
# Basic kinetic data goes here
import numpy as np
import math as m
# Activation Energies
# unit = J per mol

EA_A = 27000
EA_B = 35000
EA_C = 90000
EA_D = 63000

# Rate of reaction coefficients
# unit = 1 per s | typically variable as it depends on reaction order

k0_A = 3 * 10 ** 7
k0_B = 4 * 10 ** 8
k0_C = 5 * 10 ** 9
k0_D = 6 * 10 ** 10

R = 8.3145              # J mol^-1 K^-1

# raw kinetics array
raw_kins = np.array([[EA_A, k0_A], [EA_B, k0_B], [EA_C, k0_C], [EA_D, k0_D]])
# Kinetics array function


def k_arr(temperature):
    result = np.zeros(len(raw_kins))
    for i in range(len(raw_kins)):
        result[i] = raw_kins[i, 1] * m.e**(-raw_kins[i, 0]/R/temperature)
    return result


def reac_fun_pl(conditions, rate_constants):
    gradients = np.zeros(len(conditions))

    for i in range(len(conditions)):
        gradients[i] = rate_constants[i] * conditions

    return gradients
