import h5py
import numpy as np
from mpmath import zeta
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
from NABOPB_for_PDE import NABOPB_for_PDE

# Suppress FEniCS solver output
set_log_level(30)


def solve_pde_at_point(x, y, nx=50, ny=50):
    # Parameters
    mu = 2
    c = float(0.9 / zeta(2))

    # Frequency index functions
    def k(j):
        return int(np.floor(-0.5 + np.sqrt(0.25 + 2 * j)))

    def m1(j):
        kj = k(j)
        return j - kj * (kj + 1) // 2

    def m2(j):
        return k(j) - m1(j)

    # Define mesh and function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, "P", 1)

    # Homogeneous Dirichlet boundary condition
    bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)

    # Define the coefficient a(x, y) as UserExpression
    class CoefficientA(UserExpression):
        def __init__(self, y, c, mu, **kwargs):
            super().__init__(**kwargs)
            self.y = y
            self.c = c
            self.mu = mu

        def eval(self, values, x_):
            x1, x2 = x_
            a_val = 1.0
            for j in range(len(self.y)):
                j1 = j + 1
                freq1 = m1(j1)
                freq2 = m2(j1)
                psi_j = (
                    self.c
                    * j1 ** (-self.mu)
                    * np.cos(2 * np.pi * freq1 * x1)
                    * np.cos(2 * np.pi * freq2 * x2)
                )
                a_val += self.y[j] * psi_j
            values[0] = a_val

        def value_shape(self):
            return ()

    # Define RHS function f(x)
    f = Constant(1.0)

    # Define and solve variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    a_expr = CoefficientA(y, c, mu, degree=3)
    a_form = dot(a_expr * grad(u), grad(v)) * dx

    u_sol = Function(V)
    solve(a_form == f * v * dx, u_sol, bc)

    return u_sol(Point(*x))


def solve_pde_at_multiple_points(x_block, y, nx=50, ny=50):
    mu = 2
    c = float(0.9 / zeta(2))

    def k(j):
        return int(np.floor(-0.5 + np.sqrt(0.25 + 2 * j)))

    def m1(j):
        kj = k(j)
        return j - kj * (kj + 1) // 2

    def m2(j):
        return k(j) - m1(j)

    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, "P", 1)
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    class CoefficientA(UserExpression):
        def __init__(self, y_coeffs, c, mu, **kwargs):
            super().__init__(**kwargs)
            self.y = y_coeffs
            self.c = c
            self.mu = mu

        def eval(self, values, x):
            # IMPORTANT: FEniCS may pass x as length-3 (x,y,0)
            x1 = x[0]
            x2 = x[1]

            a_val = 1.0
            for j in range(len(self.y)):
                j1 = j + 1
                psi_j = (
                    self.c
                    * (j1 ** (-self.mu))
                    * np.cos(2 * np.pi * m1(j1) * x1)
                    * np.cos(2 * np.pi * m2(j1) * x2)
                )
                a_val += self.y[j] * psi_j

            values[0] = a_val

        def value_shape(self):
            return ()

    f = Constant(1.0)
    u = TrialFunction(V)
    v = TestFunction(V)

    a_expr = CoefficientA(y, c, mu, degree=3)
    a_form = dot(a_expr * grad(u), grad(v)) * dx

    u_sol = Function(V)
    solve(a_form == f * v * dx, u_sol, bc)

    # Evaluate at all points in the block
    B = x_block.shape[0]
    out = np.empty((B,), dtype=np.float64)
    for i in range(B):
        out[i] = u_sol(Point(float(x_block[i, 0]), float(x_block[i, 1])))

    return out.reshape(-1, 1)  # (B,1)


def solve_single(args):
    x_loc, y_coeffs = args[:2], args[2:]
    # Transform spatial variables to approximation domain
    x_loc = (x_loc + 1) / 2
    return solve_pde_at_point(x_loc, y_coeffs)


def solve_diff(var):
    # Parallel solution of the differential equation
    with Pool() as pool:
        result = pool.map(solve_single, var)

    return np.array(result, dtype=np.complex_)


def solve_multiple(args):
    var1_block, y_coeffs = args
    # var1_block comes from NABOPB_for_PDE, typically in [-1,1]^2
    # map to [0,1]^2
    var1_block = (var1_block + 1.0) / 2.0
    return solve_pde_at_multiple_points(var1_block, y_coeffs)


def solve_diff_v2(var1, var2, blocksize=1, n_cores=96):
    M = var2.shape[0]

    tasks = [
        (var1[i * blocksize : (i + 1) * blocksize, :], var2[i, :]) for i in range(M)
    ]

    with Pool(processes=n_cores) as pool:
        result = pool.map(solve_multiple, tasks)

    return np.vstack(result)  # (M*blocksize, 1)


# Set parameters
d = 22  # Total dimensions: 2 spatial dimensions + 20 parameter dimensions
s = 1000  # sparsity
pde_dims = 2  # For NABOPB_for_PDE

basis_flag = "Cheby_OMP+"  # Specifies approximation basis and CRM

gamma = {  # Define the search space Gamma
    "type": "full",
    "w": 1,
    "N": 64,
    "sgn": "non-negative",
    "superpos": 7,
}

diminc = {"type": "default"}  # Set the dimension-incremental strategy

# Compute the approximation
# result = NABOPB(solve_diff, d, basis_flag, gamma, diminc, s)
result = NABOPB_for_PDE(solve_diff_v2, pde_dims, d, basis_flag, gamma, diminc, s)

# %% Save results
import pickle, os


def build_filename(s, N, optional=""):
    return f"s{s}n{N}{optional}.pickle"


# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = "results_diff_pde"
file_path = os.path.join(folder, build_filename(s, gamma["N"], optional="OMP+"))

if not os.path.exists(folder):
    os.makedirs(folder)

with open(file_path, "wb") as file:
    pickle.dump(result, file)
