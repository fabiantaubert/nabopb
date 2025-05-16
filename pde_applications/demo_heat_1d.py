import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from multiprocessing import Pool
import os
import sys
import inspect

# Set up directory paths to import NABOPB
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from NABOPB import NABOPB

def solve_heat_at_point(x, t, a_coeffs, alpha = 0.25, nx=1000, nt=100):
    x_vals = np.linspace(0, 1, nx)
    dx = x_vals[1] - x_vals[0]

    # Define initial condition
    def initial_condition(x):
        return sum(a_coeffs[l] * np.sin((l + 1) * np.pi * x) for l in range(len(a_coeffs)))

    u0 = initial_condition(x_vals)
    u0[0] = u0[-1] = 0

    # Define differential problem
    def heat_rhs(t, u):
        dudt = np.zeros_like(u)
        dudt[1:-1] = alpha**2 * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        return dudt

    if t == 0:
        return initial_condition(x)
    
    # Set t as final time
    t_span = (0, t)
    t_eval = np.linspace(*t_span, nt)
        
    # Solve differential problem
    sol = solve_ivp(heat_rhs, t_span, u0, t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-10)
    
    # Return solution at point x at time t
    u_interp = interp1d(x_vals, sol.y[:, -1], kind='cubic', bounds_error=False, fill_value=0.0)
    return u_interp(x)

def solve_single(args):
    x_loc, t_loc, a_coeffs = args[0], args[1], args[2:]
    # Transform variable to approximation domain
    x_loc = (x_loc + 1) / 2
    t_loc = (t_loc + 1) / 2
    return solve_heat_at_point(x_loc,t_loc, a_coeffs)

def solve_heat(var):
    # Parallel solution of the differential equation
    with Pool() as pool:
        result = pool.map(solve_single, var)

    return np.array(result, dtype=np.complex_)

# Set parameters
d = 11  # Total dimensions: 1 spatial dimension + 1 time dimension + 9 parameter dimensions
s = 100  # sparsity

basis_flag = 'Cheby_mr1l'  # Specifies approximation basis and cubature method

gamma = {  # Define the search space Gamma
    'type': 'full',
    'w': 1,
    'N': 16,
    'sgn': 'non-negative'
}

diminc = {  # Set the dimension-incremental strategy
    'type': 'default'
}

# Compute the approximation
C_mr1l_result = NABOPB(solve_heat, d, basis_flag, gamma, diminc, s)

#%% Save results
import pickle, os

def build_filename(s, N, optional=''):
    return f's{s}n{N}{optional}.pickle'

# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = 'results_heat_1d'
file_path = os.path.join(folder, build_filename(s, gamma['N'], optional='')) 

if not os.path.exists(folder):
    os.makedirs(folder)
    
with open(file_path, 'wb') as file:
    pickle.dump(C_mr1l_result, file)