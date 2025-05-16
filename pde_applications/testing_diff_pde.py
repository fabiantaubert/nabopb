import numpy as np
from mpmath import zeta
import pickle
import matplotlib.pyplot as plt
from fenics import *
from multiprocessing import Pool

# Suppress FEniCS solver output
set_log_level(30)

def solve_pde(y, nx=50, ny=50):
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
    V = FunctionSpace(mesh, 'P', 1)

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
                psi_j = self.c * j1**(-self.mu) * np.cos(2 * np.pi * freq1 * x1) * np.cos(2 * np.pi * freq2 * x2)
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

    points = np.array([mesh.coordinates()[:, 0], mesh.coordinates()[:, 1]]).T
    u_vals = u_sol.compute_vertex_values(mesh)

    return points, u_vals

# Chebyshev evaluation function
def eval_cheby(x, val, index):
    res = np.zeros((x.shape[0]), dtype=np.complex_)
    scal = (2.0 ** (np.sum(index != 0, axis=1) / 2))
    for j in range(x.shape[0]):
        res[j] = np.sum(val * scal * np.prod(np.cos(np.arccos(x[j, :]) * index), axis=1))
    return res

def single_error(args):
    # Compute the relative l2-error for a single sample
    i, y, results = args

    # Solve PDE to get reference solution
    points, ex_sol = solve_pde(y[i, :])
    
    # Broadcast y[i, :] for eval_cheby
    y_temp = np.broadcast_to(y[i, :], (points.shape[0], y.shape[1]))

    # Solution by our approximation
    approx_sol = eval_cheby(np.hstack([points*2-1, y_temp]), results['val'], results['index'])

    # Compute relative L2-error
    return np.linalg.norm(ex_sol - approx_sol, 2) / np.linalg.norm(ex_sol, 2)

def parallel_error(N, results):
    # Random draws of y
    y = np.random.rand(N, 20) * 2 - 1

    # Prepare arguments for parallel processing
    args = [(i, y, results) for i in range(N)]

    # Parallel error computation
    with Pool() as pool:
        err = pool.map(single_error, args)

    return np.array(err)

# Load approximation results
file_path = 'results_diff_pde/s100n16.pickle'
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
plt.title('Boxplot of relative $L^2$ Errors for the parametric diffusion equation', fontsize=13)
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()