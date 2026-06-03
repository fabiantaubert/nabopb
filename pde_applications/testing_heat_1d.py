import numpy as np
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool


def heat_solution_closed_form(var1, a_coeffs, alpha=0.25):
    # var1: shape (M,2), with columns x and t
    x = var1[:, 0]
    t = var1[:, 1]

    L = a_coeffs.size
    l = np.arange(L)

    decay = np.exp(-(alpha**2) * (l + 1) ** 2 * np.pi**2 * t[:, None])
    sines = np.sin((l + 1) * np.pi * x[:, None])

    # Return shape (M,)
    return np.sum(a_coeffs * decay * sines, axis=1)


def single_error(args):
    i, a, results = args

    # Uniform evaluation grid (same resolution as before)
    nx = 100
    nt = 100
    x_vals = np.linspace(0, 1, nx + 1)
    t_vals = np.linspace(0, 1, nt + 1)

    X, T = np.meshgrid(x_vals, t_vals)
    points = np.stack([X.ravel(), T.ravel()], axis=-1)

    # New exact heat solution (vectorized)
    ex_sol = heat_solution_closed_form(points, a[i, :])

    # Build augmented input for Chebyshev evaluation
    a_temp = np.broadcast_to(a[i, :], (points.shape[0], a.shape[1]))
    cheby_input = np.hstack([points * 2 - 1, a_temp])

    approx_sol = eval_cheby(cheby_input, results["val"], results["index"])

    # Relative L2 error
    return np.linalg.norm(ex_sol - approx_sol) / np.linalg.norm(ex_sol)


# Chebyshev evaluation function
def eval_cheby(x, val, index):
    res = np.zeros((x.shape[0]), dtype=np.complex_)
    scal = 2.0 ** (np.sum(index != 0, axis=1) / 2)
    for j in range(x.shape[0]):
        res[j] = np.sum(
            val * scal * np.prod(np.cos(np.arccos(x[j, :]) * index), axis=1)
        )
    return res


def parallel_error_random(N, results):
    # Random draws of a
    a = np.random.rand(N, 9) * 2 - 1

    # Prepare arguments for parallel processing
    args = [(i, a, results) for i in range(N)]

    # Parallel error computation
    with Pool() as pool:
        err = pool.map(single_error, args)

    return np.array(err)


def parallel_error(a, results):
    N = a.shape[0]
    args = [(i, a, results) for i in range(N)]

    with Pool() as pool:
        err = pool.map(single_error, args)

    return np.array(err)


# Load approximation results
file_path = "results_heat_1d/s1000n64OMP+.pickle"
with open(file_path, "rb") as file:
    results = pickle.load(file)

# Run error computation
a_test = np.load("heat_testset_10000.npy")
err = parallel_error(a_test, results)
# err = parallel_error_random(10 ** 4, results)

# Plot the error
plt.figure(figsize=(8, 2.5))
plt.boxplot(
    err,
    vert=False,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="darkblue"),
    whiskerprops=dict(color="blue"),
    capprops=dict(color="blue"),
    flierprops=dict(
        markerfacecolor="blue",
        marker="o",
        markersize=3,
        linestyle="none",
        markeredgecolor="none",
    ),
)
plt.xscale("log")
plt.xlabel("Relative $L^2$ error (log scale)", fontsize=12)
plt.title("Boxplot of relative $L^2$ Errors for the 1D heat equation", fontsize=13)
plt.grid(True, axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
