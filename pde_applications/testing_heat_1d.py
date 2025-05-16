import numpy as np
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Exact solution based on series expansion
def solve_heat(a_coeffs, nx=100, nt=100, alpha = 0.25):
    x_vals = np.linspace(0, 1, nx + 1)
    t_vals = np.linspace(0, 1, nt + 1)
    grid = np.zeros((nt + 1, nx + 1))

    for n, t in enumerate(t_vals):
        for j, x in enumerate(x_vals):
            val = 0.0
            for l, a_l in enumerate(a_coeffs, start=1):
                val += a_l * np.sin(l * np.pi * x) * np.exp(-l**2 * np.pi**2 * alpha**2 * t)
            grid[n, j] = val

    return x_vals, t_vals, grid

# Chebyshev evaluation function
def eval_cheby(x, val, index):
    res = np.zeros((x.shape[0]), dtype=np.complex_)
    scal = (2.0 ** (np.sum(index != 0, axis=1) / 2))
    for j in range(x.shape[0]):
        res[j] = np.sum(val * scal * np.prod(np.cos(np.arccos(x[j, :]) * index), axis=1))
    return res

def single_error(args):
    # Compute the relative l2-error for a single sample
    i, a, results = args

    # Get exact solution and solution grid
    x_vals, t_vals, sol_grid = solve_heat(a[i, :])
    X, T = np.meshgrid(x_vals, t_vals)
    points = np.stack([X.ravel(), T.ravel()], axis=-1)
    ex_sol = sol_grid.ravel()

    # Broadcast a[i, :] for eval_cheby
    a_temp = np.broadcast_to(a[i, :], (points.shape[0], a.shape[1]))

    # Solution by our approximation
    approx_sol = eval_cheby(np.hstack([points*2-1, a_temp]), results['val'], results['index'])

    # Compute relative L2-error
    return np.linalg.norm(ex_sol - approx_sol, 2) / np.linalg.norm(ex_sol, 2)

def parallel_error(N, results):
    # Random draws of a
    a = np.random.rand(N, 9) * 2 - 1
    
    # Prepare arguments for parallel processing
    args = [(i, a, results) for i in range(N)]
    
    # Parallel error computation
    with Pool() as pool:
        err = pool.map(single_error, args)
        
    return np.array(err)

# Load approximation results
file_path = 'results_heat_1d/s100n16.pickle'
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
plt.title('Boxplot of relative $L^2$ Errors for the 1D heat equation', fontsize=13)
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()