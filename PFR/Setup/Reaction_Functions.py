import numpy as np
from PFR.Setup.Basic_Kinetics import nu_array


'''power reaction law function'''


def reac_fun_pl(t, conditions, rate_constants):
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
        temp_c = np.zeros(len(conditions))

        for species in range(len(conditions)):
            if nu_array[reaction, species] <= 0:
                temp_c[species] = conditions[species]**abs(nu_array[reaction, species])
            else:
                temp_c[species] = 1
        # print('reaction : ')
        # print(reaction)
        # print('temp_c : ')
        # print(temp_c)
        # print('product : ')
        # print(np.prod(temp_c))
        for i in range(len(conditions)):
            gradients[i] += nu_array[reaction, i] * rate_constants[reaction] * np.prod(temp_c)
        # print(gradients)

    return gradients

