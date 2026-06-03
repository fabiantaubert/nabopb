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


def solve_burgers_single(x, a_coeffs, nu=0.05, alpha=2.0, nx=1000, nt=100):
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
    def burgers_rhs(t, u):
        dudt = np.zeros_like(u)
        dudt[1:-1] = (
            -u[1:-1] * (u[2:] - u[:-2]) / (2 * dx)
            + nu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        )
        return dudt

    # Set 1 as final time
    t_span = (0, 1.0)
    t_eval = np.linspace(*t_span, nt)

    # Solve differential problem
    sol = solve_ivp(
        burgers_rhs, t_span, u0, t_eval=t_eval, method="Radau", rtol=1e-6, atol=1e-8
    )

    # Return solution at point x at time 1
    u_interp = interp1d(
        x_vals, sol.y[:, -1], kind="cubic", bounds_error=False, fill_value=0.0
    )
    return u_interp(x)


def solve_burgers_multiple(var1_block, a_coeffs, nu=0.05, nx=1000):
    # var1_block shape: (blocksize, 1)
    xs = var1_block[:, 0]

    x_vals = np.linspace(0, 1, nx)
    dx = x_vals[1] - x_vals[0]

    # --- initial condition ---
    u0 = np.zeros(nx)
    for l, a_l in enumerate(a_coeffs, start=1):
        u0 += a_l * np.sin(l * np.pi * x_vals)
    u0[0] = u0[-1] = 0.0

    # --- Burgers equation RHS ---
    def burgers_rhs(t, u):
        dudt = np.zeros_like(u)
        dudt[1:-1] = (
            -u[1:-1] * (u[2:] - u[:-2]) / (2 * dx)
            + nu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        )
        return dudt

    # --- Solve directly to t = 1 ---
    sol = solve_ivp(
        burgers_rhs,
        (0.0, 1.0),
        u0,
        t_eval=[1.0],
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )

    u_T = sol.y[:, 0]  # solution at t=1

    # interpolate for all xs in the block
    return np.interp(xs, x_vals, u_T).reshape(-1, 1)


def solve_single(args):
    x_loc, a_coeffs = args[0], args[1:]
    # Transform spatial variable to approximation domain
    x_loc = (x_loc + 1) / 2
    return solve_burgers_single(x_loc, a_coeffs)


def solve_multiple(args):
    var1_block, a_coeffs = args
    # transform block x from [-1,1] to [0,1]
    var1_block = (var1_block + 1) / 2
    return solve_burgers_multiple(var1_block, a_coeffs)


def solve_burgers_v1(var):
    # Parallel solution of the differential equation
    with Pool() as pool:
        result = pool.map(solve_single, var)

    return np.array(result, dtype=np.complex_)


def solve_burgers_v2(var1, var2, blocksize=1):
    M = var2.shape[0]
    # Parallel solution of the differential equation
    tasks = [
        (var1[i * blocksize : (i + 1) * blocksize, :], var2[i, :]) for i in range(M)
    ]
    with Pool(processes=96) as pool:
        result = pool.map(solve_multiple, tasks)

    return np.vstack(result)


# Set parameters
d = 10  # Total dimensions: 1 spatial dimension + 9 parameter dimensions
s = 1000  # sparsity
pde_dims = 1  # For NABOPB_for_PDE

basis_flag = "Cheby_OMP+"  # Specifies approximation basis and CRM

gamma = {  # Define the search space Gamma
    "type": "full",
    "w": 1,
    "N": 64,
    "sgn": "non-negative",
}

diminc = {"type": "default"}  # Set the dimension-incremental strategy

# Compute the approximation
# result = NABOPB(solve_burgers_v1, d, basis_flag, gamma, diminc, s)
result = NABOPB_for_PDE(
    solve_burgers_v2, pde_dims, d, basis_flag, gamma, diminc, s, pde_disc=9
)

# %% Save results
import pickle, os


def build_filename(s, N, optional=""):
    return f"s{s}n{N}{optional}.pickle"


# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = "results_burgers_1d"
file_path = os.path.join(folder, build_filename(s, gamma["N"], optional="OMP+"))

if not os.path.exists(folder):
    os.makedirs(folder)

with open(file_path, "wb") as file:
    pickle.dump(result, file)
