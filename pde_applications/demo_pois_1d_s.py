from pathlib import Path
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
from NABOPB_for_PDE import NABOPB_for_PDE
from cardinal_bspline import cardinal_bspline


def solve_single(args):
    index, var, tol = args

    # Define differential problem
    def bvpfcn(x, y, a, m):
        return np.vstack([np.expand_dims(y[1], 0), -f(x, a, m)])

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
    x = var[index, 0]  # Spatial variable
    a = var[index, 1:]  # B-spline coefficients

    # Transform x from [-1, 1] to [0, 1]
    x = (x + 1) / 2

    # Set initial mesh and guess
    xmesh = np.linspace(0, 1, 5)
    y_guess = guess(xmesh)

    # Solve differential problem
    sol = solve_bvp(lambda t, y: bvpfcn(t, y, a, m), bcfcn, xmesh, y_guess, tol=tol)

    # Return solution at x
    return sol.sol(x)[0]


def solve_pois1d(var, tol=1e-6):
    # Parallel solution of the differential equation
    with Pool() as pool:
        val = pool.map(solve_single, [(i, var, tol) for i in range(var.shape[0])])
    return np.array(val, dtype=np.complex128)



def solve_multiple(args):
    var1, a, tol = args

    # Solve the same BVP once for all spatial samples of one parameter vector.
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

    def bvpfcn(x, y, a, m):
        return np.vstack([np.expand_dims(y[1], 0), -f(x, a, m)])

    def bcfcn(ya, yb):
        return np.array([ya[0], yb[0]])

    def guess(x):
        return np.vstack([np.sin(x), np.cos(x)])

    m = 3
    x = (var1[:, 0] + 1) / 2
    xmesh = np.linspace(0, 1, 5)
    y_guess = guess(xmesh)

    sol = solve_bvp(
        lambda t, y: bvpfcn(t, y, a, m),
        bcfcn,
        xmesh,
        y_guess,
        tol=tol,
    )

    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    # Preserve the continuous BVP interpolant used in the standard solver.
    return sol.sol(x)[0].reshape(-1, 1)


def solve_pois1d_v2(var1, var2, blocksize=1, tol=1e-6):
    M = var2.shape[0]
    tasks = [
        (
            var1[i * blocksize : (i + 1) * blocksize, :],
            var2[i, :],
            tol,
        )
        for i in range(M)
    ]

    with Pool() as pool:
        result = pool.map(solve_multiple, tasks)

    return np.vstack(result)


# Set parameters
d = 10  # Total dimensions: 1 spatial dimension + 9 parameter dimensions
s = 1000  # sparsity
pde_dims = 1

basis_flag = "Cheby_OMP"  # Specifies approximation basis and cubature method
# basis_flag = "Cheby_OMP+"  # For PDE-aware NABOPB_for_PDE

gamma = {  # Define the search space Gamma
    "type": "full",
    "N": 64,
    "W": 1,
    "sgn": "non-negative",
}

diminc = {"type": "default"}  # Set the dimension-incremental strategy

# Compute the approximation (choose the matching basis_flag above)
result = NABOPB(
    lambda var: solve_pois1d(var, tol=1e-6), d, basis_flag, gamma, diminc, s
)
# result = NABOPB_for_PDE(solve_pois1d_v2, pde_dims, d, basis_flag, gamma, diminc, s)

# %% Save results
import pickle, os


def build_filename(s, N, optional=""):
    return f"s{s}n{N}{optional}.pickle"


# Set folder, create file path. Note: Use 'optional' to save multiple versions with same s and N.
folder = Path(__file__).resolve().parent / "results_pois_1d_s"
file_path = os.path.join(folder, build_filename(s, gamma["N"], optional=basis_flag.split("_", 1)[1]))

if not os.path.exists(folder):
    os.makedirs(folder)

with open(file_path, "wb") as file:
    pickle.dump(result, file)
