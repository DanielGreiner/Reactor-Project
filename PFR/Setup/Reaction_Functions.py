import numpy as np
from PFR.Setup.Basic_Kinetics import nu_array


'''power reaction law function'''


def reac_fun_pl(conditions, rate_constants):
    gradients = np.zeros(len(conditions))

    '''for species in range(len(conditions)):
        for reaction in range(len(nu_array)):
            temp = rate_constants[reaction] * conditions[species] * nu_array[reaction, species]
            gradients[species] = gradients[species] + temp
            #print('species = ' + str(species) + ' -> ' + str(temp))
    print('gradients: ')
    print(gradients)'''

    # reaction orders from nu
    for reaction in range(len(nu_array)):
        temp_c = np.array(len(nu_array))
        for species in range(len(conditions)):
            if nu_array[reaction, species] <= 0:
                temp_c[species] = conditions[species]**abs(nu_array[reaction, species])
            else:
                temp_c[species] = 1
            gradients[species] = nu_array[reaction, species] * rate_constants[reaction] * temp_c
    return gradients

