# Well hello there!
# Basic kinetic data goes here
import numpy as np
import math as m

'''system of reactions'''
# reaction 1:
''' A + 2B -> C '''

# reaction 2:
''' 2A -> D '''

# reaction 3:
''' C + D -> E '''

# reaction 4:
''' 2E -> B + C '''

'''activation energies'''
# unit = J per mol

EA_1 = 75000
EA_2 = 75000
EA_3 = 85000
EA_4 = 85000

'''rate of reaction coefficients'''
# unit = 1 per s | typically variable as it depends on reaction order

k0_1 = 5 * 10 ** 9
k0_2 = 5 * 10 ** 9
k0_3 = 2 * 10 ** 9
k0_4 = 2 * 10 ** 9

R = 8.3145              # J mol^-1 K^-1

# raw kinetics array
raw_kins = np.array([[EA_1, k0_1], [EA_2, k0_2], [EA_3, k0_3], [EA_4, k0_4]])

'''kinetic coefficients'''
# reaction 1:
nu_A1 = -1
nu_B1 = -2
nu_C1 = 1
nu_D1 = 0
nu_E1 = 0

# reaction 2:
nu_A2 = -2
nu_B2 = 0
nu_C2 = 0
nu_D2 = 1
nu_E2 = 0

# reaction 3:
nu_A3 = 0
nu_B3 = 0
nu_C3 = -1
nu_D3 = -1
nu_E3 = 1

# reaction 4:
nu_A4 = 0
nu_B4 = 1
nu_C4 = 1
nu_D4 = 0
nu_E4 = -2

# kinetic coefficient array

nu_array = np.array([[nu_A1, nu_B1, nu_C1, nu_D1, nu_E1], [nu_A2, nu_B2, nu_C2, nu_D2, nu_E2],
                     [nu_A3, nu_B3, nu_C3, nu_D3, nu_E3], [nu_A4, nu_B4, nu_C4, nu_D4, nu_E4]])


'''kinetics array function'''


def k_arr(temperature):
    result = np.zeros(len(raw_kins))
    for i in range(len(raw_kins)):
        result[i] = raw_kins[i, 1] * m.e**(-raw_kins[i, 0]/R/temperature)
    return result



