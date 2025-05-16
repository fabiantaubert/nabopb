import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import pickle
import os
import sys
import inspect
from multiprocessing import Pool

# Set up directory path to import cardinal B-spline
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from cardinal_bspline import cardinal_bspline

def solve_pois1d(x,a):
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
        tol=1e-9
    )
    
    # Return solution at x
    return np.interp(x, sol.x, sol.y[0])

def eval_cheby(x, val, index):
    # Evaluate the Chebyshev polynomial
    res = np.zeros((x.shape[0]),dtype=np.complex_)
    scal = 2.0 ** (np.sum(index != 0, axis=1) / 2)
    for j in range(x.shape[0]):
        res[j] = np.sum(val * scal * np.prod(np.cos(np.arccos(x[j,:]) * index), axis=1)) 
    return res

def single_error(args):
    # Compute the relative l2-error for a single sample
    i, a, x, results = args

    # Broadcast a[i, :] for eval_cheby
    a_temp = np.broadcast_to(a[i, :], (1000, a.shape[1]))

    # Solve ODE to get reference solution
    ex_sol = solve_pois1d(x, a[i, :])

    # Solution by our approximation
    approx_sol = eval_cheby(np.hstack([np.expand_dims(x, 1), a_temp]), results['val'], results['index'])

    # Compute relative l2-error
    return np.linalg.norm(ex_sol - approx_sol, 2) / np.linalg.norm(ex_sol, 2)

def parallel_error(N, results):
    # Random draws of a
    a = np.random.rand(N, 9) * 2 - 1
    
    # Set spatial points
    x = np.linspace(-1, 1, 1000)

    # Prepare arguments for parallel processing
    args = [(i, a, x, results) for i in range(N)]

    # Parallel error computation
    with Pool() as pool:
        err = pool.map(single_error, args)

    return np.array(err)

# Load approximation results
file_path = 'results_pois_1d_s/s100n16.pickle'
with open(file_path, 'rb') as file:
    results = pickle.load(file)
    
# Run error computation
err = parallel_error(10**2, results)

# Plot the error
plt.figure(figsize=(8, 2.5))
plt.boxplot(err, vert=False, widths=0.5, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='darkblue'),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'),
            flierprops=dict(markerfacecolor='blue', marker='o', markersize=3, linestyle='none', markeredgecolor='none'))
plt.xscale('log')
plt.xlabel('Relative $L^2$ error (log scale)', fontsize=12)
plt.title('Boxplot of relative $L^2$ Errors for the 1D Poisson equation', fontsize=13)
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()