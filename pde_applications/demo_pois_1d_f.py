import numpy as np
import os
import sys
import inspect

# Set up directory paths to import NABOPB
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from NABOPB import NABOPB

def solve_pois1d(var):
    # Split input array: x (spatial variable) and a (Fourier coefficients)
    x = var[:, 0]
    a = var[:, 1:]

    # Determine the index range for Fourier coefficients
    N = (a.shape[1] - 1) / 2
    if not N.is_integer():
        raise ValueError('N is not an integer.')
    N = int(N)

    ind = np.linspace(-N, N, 2 * N + 1)  # Index range from -N to N

    # Transform spatial variable x from [-1, 1] to [0, 1]
    x = (x + 1) / 2

    # Compute the solution u(x, a) using the exact formula
    with np.errstate(divide='ignore', invalid='ignore'):
        temp = a / (4 * np.pi**2 * ind**2) * (np.exp(2 * np.pi * 1j * np.expand_dims(ind, 0) * np.expand_dims(x, 1)) - 1)
    val = a[:, N] / 2 * x * (1 - x) + np.nansum(temp, axis=1)

    return val

# Set parameters
d = 10  # Total dimensions: 1 spatial dimension + 9 parameter dimensions
s = 100  # sparsity

basis_flag = 'Cheby_mr1l'  # Specifies approximation basis and cubature method

gamma = {  # Define the search space Gamma
    'type': 'full',  
    'N': 16,         
    'W': 1,          
    'sgn': 'non-negative'
}

diminc = {  # Set the dimension-incremental strategy
    'type': 'default'
}

# Compute the approximation
C_mr1l_result = NABOPB(solve_pois1d, d, basis_flag, gamma, diminc, s)

#%% Save results
import pickle, os

def build_filename(s, N, optional=''):
    return f's{s}n{N}{optional}.pickle'

# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = 'results_pois_1d_f'
file_path = os.path.join(folder, build_filename(s, gamma['N'], optional='')) 

if not os.path.exists(folder):
    os.makedirs(folder)
    
with open(file_path, 'wb') as file:
    pickle.dump(C_mr1l_result, file)