from pathlib import Path
import numpy as np
import pickle
from fenics import *
from multiprocessing import Pool

# Suppress FEniCS solver output
set_log_level(30)


def solve_pde(coef, nx=50, ny=50):
    # Define mesh and function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, "P", 1)

    # Homogeneous Dirichlet boundary condition
    bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)

    # Define frequencies
    freq = np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]])

    # Define real and imaginary parts of RHS as UserExpression
    class FourierPartialSum(UserExpression):
        def __init__(self, coef, mode="real", **kwargs):
            super().__init__(**kwargs)
            self.coef = coef
            self.mode = mode

        def eval(self, values, x_):
            x_val, y_val = x_
            s = sum(
                self.coef[k]
                * np.exp(2j * np.pi * (freq[0, k] * x_val + freq[1, k] * y_val))
                for k in range(9)
            )
            values[0] = np.real(s) if self.mode == "real" else np.imag(s)

        def value_shape(self):
            return ()

    # Define and solve variational problems
    u_re, u_im = TrialFunction(V), TrialFunction(V)
    v = TestFunction(V)

    f_re = FourierPartialSum(coef, mode="real", degree=2)
    f_im = FourierPartialSum(coef, mode="imag", degree=2)

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
    res = np.zeros((x.shape[0]), dtype=np.complex128)
    scal = 2.0 ** (np.sum(index != 0, axis=1) / 2)
    for j in range(x.shape[0]):
        res[j] = np.sum(
            val * scal * np.prod(np.cos(np.arccos(x[j, :]) * index), axis=1)
        )
    return res


def single_error(args):
    i, a, results_1000, results_2000, results_4000 = args

    # Compute the reference solution once for all three approximations.
    points, ex_sol = solve_pde(a[i, :])
    a_temp = np.broadcast_to(a[i, :], (points.shape[0], a.shape[1]))
    eval_points = np.hstack([points * 2 - 1, a_temp])

    approx_1000 = eval_cheby(
        eval_points, results_1000["val"], results_1000["index"]
    )
    approx_2000 = eval_cheby(
        eval_points, results_2000["val"], results_2000["index"]
    )
    approx_4000 = eval_cheby(
        eval_points, results_4000["val"], results_4000["index"]
    )

    norm_ex = np.linalg.norm(ex_sol, 2)

    return (
        np.linalg.norm(ex_sol - approx_1000, 2) / norm_ex,
        np.linalg.norm(ex_sol - approx_2000, 2) / norm_ex,
        np.linalg.norm(ex_sol - approx_4000, 2) / norm_ex,
    )


def parallel_error(N, results_1000, results_2000, results_4000):
    rng = np.random.default_rng(12345)
    a = rng.uniform(-1, 1, size=(N, 9))
    args = [
        (i, a, results_1000, results_2000, results_4000)
        for i in range(N)
    ]

    with Pool() as pool:
        err = pool.map(single_error, args)

    return np.asarray(err)


# Load the three approximations from the dissertation.
results_dir = Path(__file__).resolve().parent / "results_pois_2d"
with open(results_dir / "s1000n64.pickle", "rb") as file:
    results_1000 = pickle.load(file)

with open(results_dir / "s2000n128.pickle", "rb") as file:
    results_2000 = pickle.load(file)

with open(results_dir / "s4000n256.pickle", "rb") as file:
    results_4000 = pickle.load(file)

# Evaluate all approximations on the same 10,000 parameter vectors.
err = parallel_error(10**4, results_1000, results_2000, results_4000)

# Rows correspond to (1000, 64), (2000, 128), and (4000, 256).
# Columns: minimum, first quartile, median, third quartile, maximum.
boxstats = np.column_stack(
    [
        np.min(err, axis=0),
        np.percentile(err, 25, axis=0),
        np.median(err, axis=0),
        np.percentile(err, 75, axis=0),
        np.max(err, axis=0),
    ]
)

np.savetxt(
    results_dir / "pois2d_errors_t.csv",
    boxstats,
    delimiter=",",
    header="lw,lq,med,uq,uw",
    comments="",
)
