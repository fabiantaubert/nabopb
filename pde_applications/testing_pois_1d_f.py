import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool

def solve_pois1d(x,a):
    # Determine the index range for Fourier coefficients
    N = (a.shape[0] - 1) / 2
    if not N.is_integer():
        raise ValueError('N is not an integer.')
    N = int(N)

    ind = np.linspace(-N, N, 2 * N + 1)  # Index range from -N to N

    # Transform spatial variable x from [-1, 1] to [0, 1]
    x = (x + 1) / 2

    # Compute the solution u(x, a) using the exact formula
    with np.errstate(divide='ignore', invalid='ignore'):
        temp = a / (4 * np.pi**2 * ind**2) * (np.exp(2 * np.pi * 1j * np.expand_dims(ind, 0) * np.expand_dims(x, 1)) - 1)
    val = a[N] / 2 * x * (1 - x) + np.nansum(temp, axis=1)

    return val

def complex_arccos(z):
    # Complex arccos function when using coefficients a outside of [-1,1]
    return -1j * np.log(z + 1j * np.sqrt(1+0j-z**2))

def eval_cheby(x, val, index):
    # Evaluate the Chebyshev polynomial
    res = np.zeros((x.shape[0]), dtype=np.complex_)
    scal = 2.0 ** (np.sum(index != 0, axis=1) / 2)
    for j in range(x.shape[0]):
        res[j] = np.sum(val * scal * np.prod(np.cos(complex_arccos(x[j, :]) * index), axis=1))
    return res

def single_error(args):
    # Compute the relative l2-error for a single sample
    i, a, x, results = args

    # Broadcast a[i, :] for eval_cheby
    a_temp = np.broadcast_to(a[i, :], (1000, a.shape[1]))

    # Exact solution
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
file_path = 'results_pois_1d_f/s100n16.pickle'
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