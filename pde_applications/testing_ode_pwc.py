from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt


def eval_cheby(x, val, index):
    # Evaluate the Chebyshev polynomial
    res = np.zeros((x.shape[0]), dtype=np.complex128)
    scal = 2.0 ** (np.sum(index != 0, axis=1) / 2)
    for j in range(x.shape[0]):
        res[j] = np.sum(
            val * scal * np.prod(np.cos(np.arccos(x[j, :]) * index), axis=1)
        )
    return res


def eval_for_x(x):
    a_given = np.array(
        [0, 0, 0, 0, -2, -2, -2, -2]
    )  # B-spline coefficients for given RHS f
    return eval_cheby(
        np.append(x, 1 / 2 * a_given).reshape(1, -1), results["val"], results["index"]
    )


# Define the true solution
def true_solution(x):
    if -1 <= x < 0:
        return -2 / 3 * x - 2 / 3
    elif 0 <= x <= 1:
        return x**2 - 1 / 3 * x - 2 / 3
    else:
        return 0.0


def W(x, ell):
    left = -1 + ell / 4
    right = -1 + (ell + 1) / 4

    if x < left:
        return 0.0
    elif x < right:
        return (
            0.5 * x**2
            + (1 - ell / 4) * x
            + 0.5 * (-1 + ell / 4) ** 2
        )
    else:
        return 0.25 * x + 7 / 32 - ell / 16


def true_solution_general(x, a):
    S = sum(a[ell] * W(x, ell) for ell in range(8))

    C1 = (
        sum(a[ell] * (15 - 2 * ell) / 96 for ell in range(8))
        + sum(a[ell] * (7 - 2 * ell) / 96 for ell in range(4))
    )
    C2 = 2 * C1

    correction = sum(a[ell] * (7 - 2 * ell) / 32 for ell in range(4))

    if x < 0:
        return -2 * S + 2 * C1 * x + C2
    else:
        return -S + C1 * x + C2 - correction


def single_error(args):
    i, a, results = args
    x_values = np.linspace(-1, 1, 1000)

    # Map the reference parameters [-1,1]^8 to physical values [-2,2]^8.
    a_phys = 2 * a[i, :]
    ex_sol = np.array([true_solution_general(x, a_phys) for x in x_values])

    a_temp = np.broadcast_to(a[i, :], (x_values.shape[0], a.shape[1]))
    approx_sol = eval_cheby(
        np.hstack([x_values.reshape(-1, 1), a_temp]),
        results["val"],
        results["index"],
    )
    approx_sol = np.real_if_close(approx_sol)
    return np.linalg.norm(ex_sol - approx_sol, 2) / np.linalg.norm(ex_sol, 2)


def parallel_error(N, results):
    a = np.random.rand(N, 8) * 2 - 1
    args = [(i, a, results) for i in range(N)]

    with Pool() as pool:
        err = pool.map(single_error, args)

    return np.asarray(err)


# Load approximation results
file_path = Path(__file__).resolve().parent / "results_ode_pwc" / "s1000n64OMP+.pickle"
with open(file_path, "rb") as file:
    results = pickle.load(file)

# Evaluate the true solution at corresponding x-values
x_values = np.linspace(-1, 1, 1000)
true_values = np.array([true_solution(x) for x in x_values])

# Solution by our approximation
computed_values = np.array([eval_for_x(x) for x in x_values]).reshape(-1)

# Compute absolute error
abs_error = np.abs(computed_values - true_values)

# Print error statistics
print(f"Max absolute error: {np.max(abs_error)}")
print(f"Mean absolute error: {np.mean(abs_error)}")

# Plot comparison (FEniCS Solution vs True Solution)
plt.figure(figsize=(10, 6))
plt.plot(x_values, computed_values, "o", label="Our surrogate")
plt.plot(x_values, true_values, "-", label="True Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Comparison of surrogate and analytical solution")
plt.show()

# Plot absolute error
plt.figure(figsize=(10, 6))
plt.plot(x_values, abs_error, label="Absolute Error", color="r")
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.legend()
plt.title("Absolute Error")
plt.show()

# Relative errors over 10,000 parameter vectors, as in the dissertation.
err = parallel_error(10**4, results)

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
plt.xlabel(r"Relative $L^2$ error (log scale)", fontsize=12)
plt.title("Discontinuous-coefficient diffusion equation", fontsize=13)
plt.grid(True, axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
