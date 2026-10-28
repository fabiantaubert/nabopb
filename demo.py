# %% Random Trigonometric Polynomial

import numpy as np
from NABOPB import NABOPB


def coefficients_in_reference_order(result, freq):
    """Match detected coefficients to reference frequencies, independent of row order."""
    index = result["index"]
    if index.shape != freq.shape:
        return None
    result_order = np.lexsort(index.T[::-1])
    reference_order = np.lexsort(freq.T[::-1])
    if not np.array_equal(index[result_order], freq[reference_order]):
        return None
    aligned = np.empty_like(result["val"])
    aligned[reference_order] = result["val"][result_order]
    return aligned


# Set parameters and flags
d = 4
s = 100

basis_flag1 = "Fourier_r1l"
basis_flag2 = "Fourier_mr1l"

gamma = {"type": "full", "N": 16, "sgn": "default"}

diminc = {"type": "default", "workers": 0}

# Draw random coefficients
# np.random.seed(339)  # uncomment for fixed frequencies
freq = np.zeros((s, d), dtype=int)
while freq.shape[0] != len(np.unique(freq, axis=0)):
    freq = np.random.randint(-gamma["N"], gamma["N"] + 1, size=(s, d))
coef = np.zeros(s)
while len(coef) != len(np.unique(coef)):
    coef = np.sort(1 / 100000 * np.random.randint(10, 1000001, s))[::-1]

# Set f
f = lambda x: np.transpose(coef @ np.exp(2 * np.pi * 1j * freq @ x.T))

# Compute approximations
F_r1l_result = NABOPB(lambda x: f(x), d, basis_flag1, gamma, diminc, s)
F_mr1l_result = NABOPB(lambda x: f(x), d, basis_flag2, gamma, diminc, s)

# Show results
if "F_r1l_result" in locals():
    print("====Fourier_r1l====")
    matched_coef = coefficients_in_reference_order(F_r1l_result, freq)
    if matched_coef is not None:
        err2_l2 = np.linalg.norm(coef - matched_coef, 2)
        err2_inf = np.linalg.norm(coef - matched_coef, np.inf)
        print(f'Detection successful, samples: {np.sum(F_r1l_result["sample_sizes"])}')
        print(f"l_2-error: {err2_l2:e}")
        print(f"max-error: {err2_inf:e}")
    else:
        print(f'Detection failed, samples: {np.sum(F_r1l_result["sample_sizes"])}')
    print(f'Time: {F_r1l_result["run_times"][3]}')

if "F_mr1l_result" in locals():
    print("====Fourier_mr1l====")
    matched_coef = coefficients_in_reference_order(F_mr1l_result, freq)
    if matched_coef is not None:
        err4_l2 = np.linalg.norm(coef - matched_coef, 2)
        err4_inf = np.linalg.norm(coef - matched_coef, np.inf)
        print(f'Detection successful, samples: {np.sum(F_mr1l_result["sample_sizes"])}')
        print(f"l_2-error: {err4_l2:e}")
        print(f"max-error: {err4_inf:e}")
    else:
        print(f'Detection failed, samples: {np.sum(F_mr1l_result["sample_sizes"])}')
    print(f'Time: {F_mr1l_result["run_times"][3]}')

print("\n")

#%% 10d Test Function

import numpy as np
from NABOPB import NABOPB
from bspline_test_10d import bspline_test_10d, bspline_test_10d_fouriercoeff

# Set parameters and flags
d = 10
s = 100

basis_flag1 = "Fourier_mr1l"

gamma = {"type": "full", "N": 16, "sgn": "default", "superpos": 4}

diminc = {"type": "default", "workers": 0}

# Compute approximations
F_mr1l_approx_result = NABOPB(
    lambda x: bspline_test_10d(x), d, basis_flag1, gamma, diminc, s
)

# Show results
if "F_mr1l_approx_result" in locals():
    print("====Fourier_mr1l====")
    true_val3 = bspline_test_10d_fouriercoeff(F_mr1l_approx_result["index"])[0]
    err3 = {
        "l2": np.linalg.norm(true_val3 - F_mr1l_approx_result["val"], 2),
        "inf": np.linalg.norm(true_val3 - F_mr1l_approx_result["val"], np.inf),
    }
    print(f'approx. squared L2 norm: {np.linalg.norm(F_mr1l_approx_result["val"], 2)**2:.5e}')
    print(f'l_2-error: {err3["l2"]:.6e}')
    print(f'max-error: {err3["inf"]:.6e}')
    print(
        f'samples: {sum(F_mr1l_approx_result["sample_sizes"])}, candidates: {sum(F_mr1l_approx_result["cand_sizes"])}, time: {F_mr1l_approx_result["run_times"][3]:.6f}'
    )

print("\n")

#%% Random Chebyshev Polynomial

import numpy as np
from NABOPB import NABOPB

# Set parameters and flags
d = 4
s = 100

basis_flag1 = "Cheby_mr1l"
basis_flag2 = "Cheby_mr1l_subsampling"

gamma = {"type": "full", "N": 16, "sgn": "non-negative"}

diminc = {"type": "default"}

# Draw random coefficients
# np.random.seed(333)  # uncomment for fixed frequencies
freq = np.zeros((s, d), dtype=int)
while freq.shape[0] != len(np.unique(freq, axis=0)):
    freq = np.random.randint(0, gamma["N"] + 1, (s, d))
coef = np.zeros(s)
while len(coef) != len(np.unique(coef)):
    coef = np.sort(1 / 100000 * np.random.randint(10, 1000001, s))[::-1]

# Set f
def f_cheby_test(x, freq, coef):
    x = np.arccos(x)
    mat = np.zeros((freq.shape[0], x.shape[0]))
    for i in range(freq.shape[0]):
        mat[i, :] = np.prod(
            np.cos((np.ones((x.shape[0], 1)) * freq[i, :]) * x), axis=1
        ) * (2.0 ** (np.sum(freq[i, :] != 0) / 2))
    val = np.transpose(coef @ mat)
    return val


# Compute approximations
C_mr1l_result = NABOPB(lambda x: f_cheby_test(x, freq, coef), d, basis_flag1, gamma, diminc, s)
C_ssmr1l_result = NABOPB(lambda x: f_cheby_test(x, freq, coef), d, basis_flag2, gamma, diminc, s)

# Show results
if "C_mr1l_result" in locals():
    print("====Cheby_mr1l====")
    matched_coef = coefficients_in_reference_order(C_mr1l_result, freq)
    if matched_coef is not None:
        err1_l2 = np.linalg.norm(coef - matched_coef)
        err1_inf = np.max(np.abs(coef - matched_coef))
        print(f'Detection successful, samples: {np.sum(C_mr1l_result["sample_sizes"])}')
        print(f"l_2-error: {err1_l2:e}")
        print(f"max-error: {err1_inf:e}")
    else:
        print(f'Detection failed, samples: {np.sum(C_mr1l_result["sample_sizes"])}')
    print(f'Time: {C_mr1l_result["run_times"][3]}')

if "C_ssmr1l_result" in locals():
    print("====Cheby_mr1l_subsampling====")
    matched_coef = coefficients_in_reference_order(C_ssmr1l_result, freq)
    if matched_coef is not None:
        err1_l2 = np.linalg.norm(coef - matched_coef)
        err1_inf = np.max(np.abs(coef - matched_coef))
        print(
            f'Detection successful, samples: {np.sum(C_ssmr1l_result["sample_sizes"])}'
        )
        print(f"l_2-error: {err1_l2:e}")
        print(f"max-error: {err1_inf:e}")
    else:
        print(f'Detection failed, samples: {np.sum(C_ssmr1l_result["sample_sizes"])}')
    print(f'Time: {C_ssmr1l_result["run_times"][3]}')

print("\n")

#%% 9d Test Function

import numpy as np
from NABOPB import NABOPB
from bsplinet_test_9d import bsplinet_test_9d, bsplinet_test_9d_chat

# Set parameters and flags
d = 9
s = 100

basis_flag1 = "Cheby_mr1l"

gamma = {"type": "full", "w": 1, "N": 16, "sgn": "non-negative", "superpos": 5}

diminc = {"type": "default"}

# Compute approximations
C_mr1l_approx_result = NABOPB(lambda x: bsplinet_test_9d(x), d, basis_flag1, gamma, diminc, s)

# Show results

if "C_mr1l_approx_result" in locals():
    print("====Cheby_mr1l====")
    scale2 = np.sqrt(np.prod(2 ** (C_mr1l_approx_result["index"] != 0), axis=1))
    val2 = C_mr1l_approx_result["val"] * scale2
    true_val2, real_norm2 = bsplinet_test_9d_chat(C_mr1l_approx_result["index"])
    err2 = {
        "l2": np.linalg.norm(true_val2 - val2, 2),
        "inf": np.linalg.norm(true_val2 - val2, np.inf),
    }
    print(f"approx. squared L2 norm: {np.linalg.norm(C_mr1l_approx_result['val'], 2)**2:.5e}")
    print(f"reference squared L2 norm (detected support): {np.linalg.norm(true_val2 / scale2, 2)**2:.5e}")
    print(f"target squared L2 norm: {real_norm2:.5e}")
    print(f'l_2-error: {err2["l2"]:.6e}')
    print(f'max-error: {err2["inf"]:.6e}')
    print(
        f'samples: {sum(C_mr1l_approx_result["sample_sizes"])}, candidates: {sum(C_mr1l_approx_result["cand_sizes"])}, time: {C_mr1l_approx_result["run_times"][3]:.6f}'
    )

print("\n")
