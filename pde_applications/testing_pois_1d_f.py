from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
from multiprocessing import Pool


def solve_pois1d(x, a):
    # Determine the index range for Fourier coefficients
    N = (a.shape[0] - 1) / 2
    if not N.is_integer():
        raise ValueError("N is not an integer.")
    N = int(N)

    ind = np.arange(-N, N + 1)  # Index range from -N to N

    # Transform spatial variable x from [-1, 1] to [0, 1]
    x = (x + 1) / 2

    # Compute the solution u(x, a) using the exact formula
    with np.errstate(divide="ignore", invalid="ignore"):
        temp = (
            a
            / (4 * np.pi**2 * ind**2)
            * (np.exp(2j * np.pi * np.expand_dims(ind, 0) * np.expand_dims(x, 1)) - 1)
        )
    val = a[N] / 2 * x * (1 - x) + np.nansum(temp, axis=1)

    return val


def complex_arccos(z):
    # Complex arccos function when using coefficients a outside of [-1,1]
    return -1j * np.log(z + 1j * np.sqrt(1 + 0j - z**2))


def eval_cheby(x, val, index):
    # Evaluate the Chebyshev polynomial
    res = np.zeros((x.shape[0]), dtype=np.complex128)
    scal = 2.0 ** (np.sum(index != 0, axis=1) / 2)
    for j in range(x.shape[0]):
        res[j] = np.sum(
            val * scal * np.prod(np.cos(complex_arccos(x[j, :]) * index), axis=1)
        )
    return res


def single_error(args):
    # Compute the relative l2-error for a single sample
    i, a, x, results = args

    # Broadcast a[i, :] for eval_cheby
    a_temp = np.broadcast_to(a[i, :], (1000, a.shape[1]))

    # Exact solution
    ex_sol = solve_pois1d(x, a[i, :])

    # Solution by our approximation
    approx_sol = eval_cheby(
        np.hstack([np.expand_dims(x, 1), a_temp]), results["val"], results["index"]
    )

    # Compute relative l2-error
    return np.linalg.norm(ex_sol - approx_sol, 2) / np.linalg.norm(ex_sol, 2)


def parallel_error(N, results, real_bound=1.0, imag_bound=0.0):
    # Draw parameter vectors in the specified real or complex test domain.
    a = (np.random.rand(N, 9) * 2 - 1) * real_bound
    if imag_bound > 0:
        a = a + 1j * (np.random.rand(N, 9) * 2 - 1) * imag_bound

    x = np.linspace(-1, 1, 1000)
    args = [(i, a, x, results) for i in range(N)]

    with Pool() as pool:
        err = pool.map(single_error, args)

    return np.asarray(err)


# Load approximation results
file_path = Path(__file__).resolve().parent / "results_pois_1d_f" / "s1000n64test.pickle"
with open(file_path, "rb") as file:
    results = pickle.load(file)

# Reproduce the five domains from the dissertation (10,000 draws each).
np.random.seed(42)
test_domains = [
    (1, 0),  # [-1,1]^9
    (2, 0),  # [-2,2]^9
    (4, 0),  # [-4,4]^9
    (1, 1),  # [-1,1]^9 + i[-1,1]^9
    (2, 2),  # [-2,2]^9 + i[-2,2]^9
]
errs = [
    parallel_error(10**4, results, real_bound, imag_bound)
    for real_bound, imag_bound in test_domains
]

# Display the five domain-specific error distributions; do not write output files.
domain_labels = [
    r"$[-1,1]^9$",
    r"$[-2,2]^9$",
    r"$[-4,4]^9$",
    r"$[-1,1]^9+i[-1,1]^9$",
    r"$[-2,2]^9+i[-2,2]^9$",
]

plt.figure(figsize=(9, 4))
plt.boxplot(
    errs,
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
plt.yticks(range(1, len(domain_labels) + 1), domain_labels)
plt.xscale("log")
plt.xlabel(r"Relative $L^2$ error (log scale)", fontsize=12)
plt.title("1D Poisson equation: Fourier-parametrized right-hand side", fontsize=13)
plt.grid(True, axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
