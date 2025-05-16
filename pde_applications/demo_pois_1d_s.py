import numpy as np
from scipy.integrate import solve_bvp
from multiprocessing import Pool
import os
import sys
import inspect

# Set up directory paths to import NABOPB and cardinal B-spline
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from NABOPB import NABOPB
from cardinal_bspline import cardinal_bspline

def solve_single(args):
    index, var, tol = args
    # Define differential problem
    def bvpfcn(x, y, a, m):
        return np.vstack([
            np.expand_dims(y[1], 0),
            -f(x, a, m)
        ])
    # Define boundary conditions
    def bcfcn(ya, yb):
        return np.array([ya[0], yb[0]])
    # Define initial guess
    def guess(x):
        return np.vstack([np.sin(x), np.cos(x)])
    # Define RHS as sum of B-splines
    def f(x, a, m):
        if a.ndim < 2:
            a = np.expand_dims(a, 0)

        N = a.shape[1]
        val = np.zeros((a.shape[0], x.shape[0]))

        for i in range(a.shape[0]):
            for j in range(N):
                shift = (m - 2) / 2 - j
                scale = (N + 1 - m) / 1
                val[i, :] += a[i, j] * cardinal_bspline(x * scale + shift, m)

        return val
    
    m = 3  # B-spline order
    x = var[index, 0]         # Spatial variable
    a = var[index, 1:]        # B-spline coefficients

    # Transform x from [-1, 1] to [0, 1]
    x = (x + 1) / 2

    # Set initial mesh and guess
    xmesh = np.linspace(0, 1, 5)
    y_guess = guess(xmesh)

    # Solve differential problem
    sol = solve_bvp(
        lambda t, y: bvpfcn(t, y, a, m),
        bcfcn,
        xmesh,
        y_guess,
        tol=tol
    )

    # Return solution at x
    return np.interp(x, sol.x, sol.y[0])

def solve_pois1d(var, tol=1e-6):
    # Parallel solution of the differential equation
    with Pool() as pool:
        val = pool.map(solve_single, [(i, var, tol) for i in range(var.shape[0])])
    return np.array(val, dtype=np.complex_)

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
C_mr1l_result = NABOPB(lambda var: solve_pois1d(var, tol=1e-6), d, basis_flag, gamma, diminc, s)

#%% Save results
import pickle, os

def build_filename(s, N, optional=''):
    return f's{s}n{N}{optional}.pickle'

# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = 'results_pois_1d_s'
file_path = os.path.join(folder, build_filename(s, gamma['N'], optional='')) 

if not os.path.exists(folder):
    os.makedirs(folder)
    
with open(file_path, 'wb') as file:
    pickle.dump(C_mr1l_result, file)