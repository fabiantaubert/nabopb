import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import inspect
from multiprocessing import Pool
from scipy.integrate import solve_ivp

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

def solve_burgers(a_coeffs, nu=0.05, alpha=2.0, nx=100):
    x_vals = np.linspace(0, 1, nx)
    dx = x_vals[1] - x_vals[0]

    # Define initial condition
    def initial_condition(x):
        return sum(a_coeffs[l] * np.sin((l + 1) * np.pi * x) for l in range(len(a_coeffs)))

    u0 = initial_condition(x_vals)
    u0[0] = u0[-1] = 0

    # Define differential problem
    def burgers_rhs(t, u):
        dudt = np.zeros_like(u)
        dudt[1:-1] = -u[1:-1] * (u[2:] - u[:-2]) / (2 * dx) + nu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        return dudt

    # Solve differential problem
    sol = solve_ivp(burgers_rhs, [0, 1.0], u0, t_eval=[1.0], method='Radau', rtol=1e-6, atol=1e-8)

    return x_vals, sol.y[:,0]

# Exact solution at t = 1
def exact_solution(nx=1000, alpha=2.0, nu=0.05):
    x_vals = np.linspace(0, 1, nx + 1)
    u_vals = (2 * np.pi * nu * np.sin(np.pi * x_vals) * np.exp(-np.pi**2 * nu)) / \
             (alpha + np.cos(np.pi * x_vals) * np.exp(-np.pi**2 * nu))
    return x_vals, u_vals

# Exact initial condition at t=0
def u_exact_0(x):
    return (2 * np.pi * 0.05 * np.sin(np.pi * x)) / (2 + np.cos(np.pi * x))

# Project initial condition to sine basis
def project_to_sine_basis(u0, x, N=9):
    a = np.zeros(N)
    for n in range(1, N+1):
        phi_n = np.sin(n * np.pi * x)
        a[n-1] = 2 * np.dot(u0, phi_n) / 10**7
    return a

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
    
    # Solve PDE to get reference solution
    x_vals, u_true_grid = solve_burgers(a[i], nx=100)
    points = np.array([[x] for x in x_vals])
    ex_sol = u_true_grid.flatten()

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
file_path = 'results_burgers_1d/s100n16v3.pickle'
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
plt.title('Boxplot of relative $L^2$ Errors for the 1D Burgers equation', fontsize=13)
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Get sine coefficients of exact initial condition
x_vals_precise = np.linspace(0, 1, 10**7)
u_init_exact = u_exact_0(x_vals_precise)
a = project_to_sine_basis(u_init_exact, x_vals_precise)

# Evaluate exact solution at t = 1
x_vals, exact = exact_solution()

# Evaluate our approximation
points = x_vals.reshape(-1, 1)
a_temp = np.broadcast_to(a, (points.shape[0], a.shape[0]))
approx = eval_cheby(np.hstack([points*2-1, a_temp]), results['val'], results['index']).real

# Plot: Exact vs Surrogate
plt.figure(figsize=(8, 4))
plt.plot(x_vals, exact, label='Exact', lw=2, color='black')
plt.plot(x_vals, approx, '--', label='Approximation', lw=2, color='blue')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$u(x, t=1)$', fontsize=12)
plt.title('Comparison of exact solution and approximation at $t=1$', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot: Absolute Error
plt.figure(figsize=(8, 4))
plt.plot(x_vals, np.abs(exact - approx), label='Absolute Error', lw=2, color='crimson')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$|u_{\mathrm{exact}} - u_{\mathrm{approx}}|$', fontsize=12)
plt.title('Absolute Error between exact solution and approximation at $t=1$', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()