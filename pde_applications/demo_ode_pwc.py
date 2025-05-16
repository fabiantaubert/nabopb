import numpy as np
from fenics import *
import os
import sys
import inspect
from multiprocessing import Pool

# Set up directory paths to import NABOPB
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from NABOPB import NABOPB

# Suppress FEniCS solver output
set_log_level(30)

def create_solver(a_coeffs):
    # Define mesh and function space
    mesh = IntervalMesh(100, -1, 1)
    V = FunctionSpace(mesh, "CG", 1)

    # Define piece-wise constant coefficient a(x)
    class Coefficient(UserExpression):
        def eval(self, value, x):
            value[0] = 0.5 if x[0] < 0 else 1.0
        def value_shape(self):
            return ()

    a = Coefficient(degree=0)

    # Define piece-wise constant right-hand side f(x)
    class Source(UserExpression):
        def __init__(self, a_coeffs, **kwargs):
            super().__init__(**kwargs)
            self.a_coeffs = a_coeffs

        def eval(self, value, x):
            # Determine interval (each of length 1/4)
            idx = int((x[0] + 1) // 0.25)
            idx = min(idx, len(self.a_coeffs) - 1)  # Ensure within bounds
            value[0] = self.a_coeffs[idx]

        def value_shape(self):
            return ()

    f = Source(2*a_coeffs, degree=0) # Scale by 2 to match desired range

    # Homogeneous Dirichlet boundary condition
    bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)

    return V, a, f, bc

def solve_single(args):
    x, a_coeffs = args[0], args[1:]
    V, a, f, bc = create_solver(a_coeffs)

    u = TrialFunction(V)
    v = TestFunction(V)

    a_form = a * inner(grad(u), grad(v)) * dx
    L_form = f * v * dx

    u = Function(V)
    solve(a_form == L_form, u, bc)

    return u(x)

def solve_ode_pwc(var):
    # Parallel solution of the differential equation
    with Pool() as pool:
        val = pool.map(solve_single, var)
    return np.array(val, dtype=np.float64)

# Set parameters
d = 9 # Total dimensions: 1 spatial dimension + 8 parameter dimensions
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
C_mr1l_result = NABOPB(lambda var: solve_ode_pwc(var), d, basis_flag, gamma, diminc, s)

#%% Save results
import pickle, os

def build_filename(s, N, optional=''):
    return f's{s}n{N}{optional}.pickle'

# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = 'results_ode_pwc'
file_path = os.path.join(folder, build_filename(s, gamma['N'], optional='')) 

if not os.path.exists(folder):
    os.makedirs(folder)
    
with open(file_path, 'wb') as file:
    pickle.dump(C_mr1l_result, file)
