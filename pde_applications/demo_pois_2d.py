import numpy as np
from fenics import *
from multiprocessing import Pool
import os
import sys
import inspect

# Set up directory paths to import NABOPB
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from NABOPB import NABOPB

# Suppress FEniCS solver output
set_log_level(30)

def solve_pde_at_point(x, y, coef, nx=50, ny=50):
    # Define mesh and function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)

    # Homogeneous Dirichlet boundary condition
    bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)

    # Define frequencies
    freq = np.array([[-1, -1, -1,  0, 0, 0,  1, 1, 1],
                     [-1,  0,  1, -1, 0, 1, -1, 0, 1]])

    # Define real and imaginary parts of RHS as UserExpression
    class FourierPartialSum(UserExpression):
        def __init__(self, coef, mode='real', **kwargs):
            super().__init__(**kwargs)
            self.coef = coef
            self.mode = mode

        def eval(self, values, x_):
            x_val, y_val = x_
            s = sum(
                self.coef[k] * np.exp(2j * np.pi * (freq[0, k] * x_val + freq[1, k] * y_val))
                for k in range(9)
            )
            values[0] = np.real(s) if self.mode == 'real' else np.imag(s)

        def value_shape(self):
            return ()
        
    # Define and solve variational problems
    u_re, u_im = TrialFunction(V), TrialFunction(V)
    v = TestFunction(V)

    f_re = FourierPartialSum(coef, mode='real', degree=2)
    f_im = FourierPartialSum(coef, mode='imag', degree=2)

    a_re = dot(grad(u_re), grad(v)) * dx
    a_im = dot(grad(u_im), grad(v)) * dx
    
    u_re_sol = Function(V)
    solve(a_re == f_re * v * dx, u_re_sol, bc)

    u_im_sol = Function(V)
    solve(a_im == f_im * v * dx, u_im_sol, bc)

    return u_re_sol(Point(x, y)) + 1j * u_im_sol(Point(x, y))

def solve_single(args):
    x_loc, y_loc, a_coeffs = args[0], args[1], args[2:]
    # Transform spatial variables to approximation domain
    x_loc = (x_loc + 1) / 2
    y_loc = (y_loc + 1) / 2
    return solve_pde_at_point(x_loc, y_loc, a_coeffs)

def solve_pois2d(var):
    # Parallel solution of the differential equation
    with Pool() as pool:
        result = pool.map(solve_single, var)

    return np.array(result, dtype=np.complex_)

# Set parameters
d = 11  # Total dimensions: 2 spatial dimensions + 9 parameter dimensions
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
C_mr1l_result = NABOPB(solve_pois2d, d, basis_flag, gamma, diminc, s)

#%% Save results
import pickle, os

def build_filename(s, N, optional=''):
    return f's{s}n{N}{optional}.pickle'

# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = 'results_pois_2d'
file_path = os.path.join(folder, build_filename(s, gamma['N'], optional='')) 

if not os.path.exists(folder):
    os.makedirs(folder)
    
with open(file_path, 'wb') as file:
    pickle.dump(C_mr1l_result, file)