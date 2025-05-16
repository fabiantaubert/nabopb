import numpy as np
import pickle
import matplotlib.pyplot as plt
from fenics import *
from multiprocessing import Pool

# Suppress FEniCS solver output
set_log_level(30)

def solve_pde(coef, nx=50, ny=50):
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

    points = np.array([mesh.coordinates()[:, 0], mesh.coordinates()[:, 1]]).T
    u_real_vals = u_re_sol.compute_vertex_values(mesh)
    u_imag_vals = u_im_sol.compute_vertex_values(mesh)

    return points, u_real_vals + 1j * u_imag_vals

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
    points, ex_sol = solve_pde(a[i, :])
    
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
file_path = 'results_pois_2d/s100n16.pickle'
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
plt.title('Boxplot of relative $L^2$ Errors for the 2D Poisson equation', fontsize=13)
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()