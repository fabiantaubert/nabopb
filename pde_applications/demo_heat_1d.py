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
from NABOPB_for_PDE import NABOPB_for_PDE


def solve_heat_at_single(x, t, a_coeffs, alpha=0.25, nx=1000, nt=100):
    x_vals = np.linspace(0, 1, nx)
    dx = x_vals[1] - x_vals[0]

    # Define initial condition
    def initial_condition(x):
        return sum(
            a_coeffs[l] * np.sin((l + 1) * np.pi * x) for l in range(len(a_coeffs))
        )

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
    sol = solve_ivp(
        heat_rhs, t_span, u0, t_eval=t_eval, method="Radau", rtol=1e-8, atol=1e-10
    )

    # Return solution at point x at time t
    u_interp = interp1d(
        x_vals, sol.y[:, -1], kind="cubic", bounds_error=False, fill_value=0.0
    )
    return u_interp(x)


def solve_heat_at_multiple(var1, a_coeffs, alpha=0.25, nx=1000, rtol=1e-8, atol=1e-10):
    # Extract x and t arrays
    x_query = var1[:, 0]
    t_query = var1[:, 1]

    # Build spatial grid
    x_vals = np.linspace(0, 1, nx)
    dx = x_vals[1] - x_vals[0]

    # Initial condition u0(x)
    u0 = np.zeros(nx)
    for l, a_l in enumerate(a_coeffs, start=1):
        u0 += a_l * np.sin(l * np.pi * x_vals)

    # Enforce homogeneous Dirichlet
    u0[0] = u0[-1] = 0.0

    # ---- Solve PDE only once ----
    t_unique = np.unique(t_query)  # all times where we need the solution
    t_unique = np.sort(t_unique)

    # ODE RHS
    def heat_rhs(t, u):
        dudt = np.zeros_like(u)
        dudt[1:-1] = alpha**2 * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        return dudt

    # solve only up to max required time
    sol = solve_ivp(
        heat_rhs,
        t_span=(0.0, t_unique[-1]),
        y0=u0,
        t_eval=t_unique,
        method="Radau",
        rtol=rtol,
        atol=atol,
    )

    # ---- Evaluate solution at (x_i, t_i) ----
    # Precompute spatial interpolators for each time in t_unique
    spatial_interps = [
        interp1d(x_vals, sol.y[:, k], kind="cubic", bounds_error=False, fill_value=0.0)
        for k in range(len(t_unique))
    ]

    # For each query time t_i, find its index in t_unique
    time_index = np.searchsorted(t_unique, t_query)

    # Now evaluate u(x_i, t_i)
    u_out = np.zeros(len(x_query))
    for i in range(len(x_query)):
        u_out[i] = spatial_interps[time_index[i]](x_query[i])

    return u_out.reshape(-1, 1)


def solve_single(args):
    x_loc, t_loc, a_coeffs = args[0], args[1], args[2:]
    # Transform variable to approximation domain
    x_loc = (x_loc + 1) / 2
    t_loc = (t_loc + 1) / 2
    return solve_heat_at_single(x_loc, t_loc, a_coeffs)


def solve_multiple(args):
    var1, var2 = args
    # Transform variable to approximation domain
    var1 = (var1 + 1) / 2
    return solve_heat_at_multiple(var1, var2)


def solve_heat_v1(var):
    # Parallel solution of the differential equation
    with Pool(processes=96) as pool:
        result = pool.map(solve_single, var)

    return np.array(result, dtype=np.complex_)


def solve_heat_v2(var1, var2, blocksize=1):
    M = var2.shape[0]
    # Parallel solution of the differential equation
    tasks = [
        (var1[i * blocksize : (i + 1) * blocksize, :], var2[i, :]) for i in range(M)
    ]
    with Pool(processes=96) as pool:
        result = pool.map(solve_multiple, tasks)

    return np.vstack(result)


# Set parameters
d = 11  # Total dimensions: 1 spatial dimension + 1 time dimension + 9 parameter dimensions
s = 1000  # sparsity
pde_dims = 2  # For NABOPB_for_PDE

basis_flag = "Cheby_OMP+"  # Specifies approximation basis and CRM

gamma = {  # Define the search space Gamma
    "type": "full",
    "w": 1,
    "N": 64,
    "sgn": "non-negative",
}

diminc = {"type": "default"}  # Set the dimension-incremental strategy

# Compute the approximation

# result = NABOPB(solve_heat_v1, d, basis_flag, gamma, diminc, s)
result = NABOPB_for_PDE(solve_heat_v2, pde_dims, d, basis_flag, gamma, diminc, s)

# %% Save results
import pickle, os


def build_filename(s, N, optional=""):
    return f"s{s}n{N}{optional}.pickle"


# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = "results_heat_1d"
file_path = os.path.join(folder, build_filename(s, gamma["N"], optional="OMP+"))

if not os.path.exists(folder):
    os.makedirs(folder)

with open(file_path, "wb") as file:
    pickle.dump(result, file)
