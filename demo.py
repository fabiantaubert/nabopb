# %% Random Trigonometric Polynomial

import numpy as np
from NABOPB import NABOPB

# Set parameters and flags
d = 10
s = 100

basis_flag1 = 'Fourier_rand'
basis_flag2 = 'Fourier_r1l'
basis_flag3 = 'Fourier_ssr1l'
basis_flag4 = 'Fourier_mr1l'

gamma = {
    'type': 'full',
    'N': 8,
    'sgn': 'default'
}

diminc = {
    'type': 'default',
    'workers': 0
}  

# Draw random coefficients
# np.random.seed(339)  # uncomment for fixed frequencies
freq = np.zeros((s, d), dtype=int)
while freq.shape[0] != len(np.unique(freq, axis=0)):
    freq = np.random.randint(-gamma['N'], gamma['N'] + 1, size=(s, d))
coef = np.zeros(s)
while len(coef) != len(np.unique(coef)):
    coef = np.sort(1 / 100000 * np.random.randint(10, 1000001, s))[::-1]

# Set f
f = lambda x: np.transpose(coef @ np.exp(2 * np.pi * 1j * freq @ x.T))

# Compute approximations (Comment unwanted computations)
F_rand_result = NABOPB(lambda x: f(x), d, basis_flag1, gamma, diminc, s)
F_r1l_result = NABOPB(lambda x: f(x), d, basis_flag2, gamma, diminc, s)
F_ssr1l_result = NABOPB(lambda x: f(x), d, basis_flag3, gamma, diminc, s)
F_mr1l_result = NABOPB(lambda x: f(x), d, basis_flag4, gamma, diminc, s)

# Show results
if 'F_rand_result' in locals():
    print('====Variant 1====')
    if np.array_equal(F_rand_result['index'], freq):
        err1_l2 = np.linalg.norm(np.transpose(coef) - F_rand_result['val'], 2)
        err1_inf = np.linalg.norm(np.transpose(coef) - F_rand_result['val'], np.inf)
        print(f'Detection successful, samples: {np.sum(F_rand_result["sample_sizes"])}')
        print(f'l_2-error: {err1_l2:e}')
        print(f'max-error: {err1_inf:e}')
    else:
        print(f'Detection failed, samples: {np.sum(F_rand_result["sample_sizes"])}')
    print(f'Time: {F_rand_result["run_times"][3]}')

if 'F_r1l_result' in locals():
    print('====Variant 2====')
    if np.array_equal(F_r1l_result['index'], freq):
        err2_l2 = np.linalg.norm(np.transpose(coef) - F_r1l_result['val'], 2)
        err2_inf = np.linalg.norm(np.transpose(coef) - F_r1l_result['val'], np.inf)
        print(f'Detection successful, samples: {np.sum(F_r1l_result["sample_sizes"])}')
        print(f'l_2-error: {err2_l2:e}')
        print(f'max-error: {err2_inf:e}')
    else:
        print(f'Detection failed, samples: {np.sum(F_r1l_result["sample_sizes"])}')
    print(f'Time: {F_r1l_result["run_times"][3]}')

if 'F_ssr1l_result' in locals():
    print('====Variant 3====')
    if np.array_equal(F_ssr1l_result['index'], freq):
        err3_l2 = np.linalg.norm(np.transpose(coef) - F_ssr1l_result['val'], 2)
        err3_inf = np.linalg.norm(np.transpose(coef) - F_ssr1l_result['val'], np.inf)
        print(f'Detection successful, samples: {np.sum(F_ssr1l_result["sample_sizes"])}')
        print(f'l_2-error: {err3_l2:e}')
        print(f'max-error: {err3_inf:e}')
    else:
        print(f'Detection failed, samples: {np.sum(F_ssr1l_result["sample_sizes"])}')
    print(f'Time: {F_ssr1l_result["run_times"][3]}')

if 'F_mr1l_result' in locals():
    print('====Variant 4====')
    if np.array_equal(F_mr1l_result['index'], freq):
        err4_l2 = np.linalg.norm(np.transpose(coef) - F_mr1l_result['val'], 2)
        err4_inf = np.linalg.norm(np.transpose(coef) - F_mr1l_result['val'], np.inf)
        print(f'Detection successful, samples: {np.sum(F_mr1l_result["sample_sizes"])}')
        print(f'l_2-error: {err4_l2:e}')
        print(f'max-error: {err4_inf:e}')
    else:
        print(f'Detection failed, samples: {np.sum(F_mr1l_result["sample_sizes"])}')
    print(f'Time: {F_mr1l_result["run_times"][3]}')

print('\n')

#%% 10d Test Function

import numpy as np
from NABOPB import NABOPB
from bspline_test_10d import bspline_test_10d, bspline_test_10d_fouriercoeff

# Set parameters and flags
d = 10
s = 100

basis_flag1 = 'Fourier_r1l'
basis_flag2 = 'Fourier_ssr1l'
basis_flag3 = 'Fourier_mr1l'

gamma = {
    'type': 'full',
    'N': 8,
    'sgn': 'default',
    'superpos': 4
}

diminc = {
    'type': 'default',
    'workers': 0
}

# Compute approximations (Comment unwanted computations)
F_r1l_approx_result = NABOPB(lambda x: bspline_test_10d(x), d, basis_flag1, gamma, diminc, s)
F_ssr1l_approx_result = NABOPB(lambda x: bspline_test_10d(x), d, basis_flag2, gamma, diminc, s)
F_mr1l_approx_result = NABOPB(lambda x: bspline_test_10d(x), d, basis_flag3, gamma, diminc, s)

# Show results
if 'F_r1l_approx_result' in locals():
    print('====Variant 1====')
    true_val1 = bspline_test_10d_fouriercoeff(F_r1l_approx_result['index'])[0]
    err1 = {
        'l2': np.linalg.norm(true_val1 - F_r1l_approx_result['val'], 2),
        'inf': np.linalg.norm(true_val1 - F_r1l_approx_result['val'], np.inf)
    }
    print(f'abs. variance: {np.linalg.norm(F_r1l_approx_result["val"], 2)**2:.5e}')
    print(f'l_2-error: {err1["l2"]:.6e}')
    print(f'max-error: {err1["inf"]:.6e}')
    print(f'samples: {sum(F_r1l_approx_result["sample_sizes"])}, candidates: {sum(F_r1l_approx_result["cand_sizes"])}, time: {F_r1l_approx_result["run_times"][3]:.6f}')

if 'F_ssr1l_approx_result' in locals():
    print('====Variant 2====')
    true_val2 = bspline_test_10d_fouriercoeff(F_ssr1l_approx_result['index'])[0]
    err2 = {
        'l2': np.linalg.norm(true_val2 - F_ssr1l_approx_result['val'], 2),
        'inf': np.linalg.norm(true_val2 - F_ssr1l_approx_result['val'], np.inf)
    }
    print(f'abs. variance: {np.linalg.norm(F_ssr1l_approx_result["val"], 2)**2:.5e}')
    print(f'l_2-error: {err2["l2"]:.6e}')
    print(f'max-error: {err2["inf"]:.6e}')
    print(f'samples: {sum(F_ssr1l_approx_result["sample_sizes"])}, candidates: {sum(F_ssr1l_approx_result["cand_sizes"])}, time: {F_ssr1l_approx_result["run_times"][3]:.6f}')

if 'F_mr1l_approx_result' in locals():
    print('====Variant 3====')
    true_val3 = bspline_test_10d_fouriercoeff(F_mr1l_approx_result['index'])[0]
    err3 = {
        'l2': np.linalg.norm(true_val3 - F_mr1l_approx_result['val'], 2),
        'inf': np.linalg.norm(true_val3 - F_mr1l_approx_result['val'], np.inf)
    }
    print(f'abs. variance: {np.linalg.norm(F_mr1l_approx_result["val"], 2)**2:.5e}')
    print(f'l_2-error: {err3["l2"]:.6e}')
    print(f'max-error: {err3["inf"]:.6e}')
    print(f'samples: {sum(F_mr1l_approx_result["sample_sizes"])}, candidates: {sum(F_mr1l_approx_result["cand_sizes"])}, time: {F_mr1l_approx_result["run_times"][3]:.6f}')

print('\n')

#%% Random Chebyshev Polynomial

import numpy as np
from NABOPB import NABOPB

# Set parameters and flags
d = 10
s = 100

basis_flag1 = 'Cheby_rand'
basis_flag2 = 'Cheby_mr1l'
basis_flag3 = 'Cheby_mr1l_subsampling'

gamma = {
    'type': 'full',
    'N': 16,
    'sgn': 'non-negative'
}

diminc = {
    'type': 'default'
}

# Draw random coefficients
# np.random.seed(333)  # uncomment for fixed frequencies
freq = np.zeros((s, d), dtype=int)
while freq.shape[0] != len(np.unique(freq, axis=0)):
    freq = np.random.randint(0, gamma['N'] + 1, (s, d))
coef = np.zeros(s)
while len(coef) != len(np.unique(coef)):
    coef = np.sort(1 / 100000 * np.random.randint(10, 1000001, s))[::-1]
    
# Set f
def f_cheby_test(x, freq, coef):
    x = np.arccos(x)
    mat = np.zeros((freq.shape[0], x.shape[0]))
    for i in range(freq.shape[0]):
        mat[i, :] = np.prod(np.cos((np.ones((x.shape[0], 1)) * freq[i, :]) * x), axis=1) * (2.0 ** (np.sum(freq[i,:] != 0) / 2))
    val = np.transpose(coef @ mat)
    return val    
    
# Compute approximations (Comment unwanted computations)
C_rand_result = NABOPB(lambda x: f_cheby_test(x, freq, coef), d, basis_flag1, gamma, diminc, s)
C_mr1l_result = NABOPB(lambda x: f_cheby_test(x, freq, coef), d, basis_flag2, gamma, diminc, s)
C_ssmr1l_result = NABOPB(lambda x: f_cheby_test(x, freq, coef), d, basis_flag3, gamma, diminc, s)

# Show results
if 'C_rand_result' in locals():
    print('====Variant 1====')
    if np.array_equal(C_rand_result['index'], freq):
        err1_l2 = np.linalg.norm(coef - C_rand_result['val'])
        err1_inf = np.max(np.abs(coef - C_rand_result['val']))
        print(f'Detection successful, samples: {np.sum(C_rand_result["sample_sizes"])}')
        print(f'l_2-error: {err1_l2:e}')
        print(f'max-error: {err1_inf:e}')
    else:
        print(f'Detection failed, samples: {np.sum(C_rand_result["sample_sizes"])}')
    print(f'Time: {C_rand_result["run_times"][3]}')
    
if 'C_mr1l_result' in locals():
    print('====Variant 2====')
    if np.array_equal(C_mr1l_result['index'], freq):
        err1_l2 = np.linalg.norm(coef - C_mr1l_result['val'])
        err1_inf = np.max(np.abs(coef - C_mr1l_result['val']))
        print(f'Detection successful, samples: {np.sum(C_mr1l_result["sample_sizes"])}')
        print(f'l_2-error: {err1_l2:e}')
        print(f'max-error: {err1_inf:e}')
    else:
        print(f'Detection failed, samples: {np.sum(C_mr1l_result["sample_sizes"])}')
    print(f'Time: {C_mr1l_result["run_times"][3]}')

if 'C_ssmr1l_result' in locals():
    print('====Variant 3====')
    if np.array_equal(C_ssmr1l_result['index'], freq):
        err1_l2 = np.linalg.norm(coef - C_ssmr1l_result['val'])
        err1_inf = np.max(np.abs(coef - C_ssmr1l_result['val']))
        print(f'Detection successful, samples: {np.sum(C_ssmr1l_result["sample_sizes"])}')
        print(f'l_2-error: {err1_l2:e}')
        print(f'max-error: {err1_inf:e}')
    else:
        print(f'Detection failed, samples: {np.sum(C_ssmr1l_result["sample_sizes"])}')
    print(f'Time: {C_ssmr1l_result["run_times"][3]}')

print('\n')

#%% 9d Test Function

import numpy as np
from NABOPB import NABOPB
from bsplinet_test_9d import bsplinet_test_9d, bsplinet_test_9d_chat

# Set parameters and flags
d = 9
s = 100

basis_flag1 = 'Cheby_rand'
basis_flag2 = 'Cheby_mr1l'
basis_flag3 = 'Cheby_mr1l_subsampling'

gamma = {
    'type': 'full',
    'w': 1,
    'N': 16,
    'sgn': 'non-negative',
    'superpos': 5
}

diminc = {
    'type': 'default',
    'workers': 4  # only for dyadic
}

# Compute approximations (Comment unwanted computations)
C_rand_approx_result = NABOPB(lambda x: bsplinet_test_9d(x), d, basis_flag1, gamma, diminc, s)
C_mr1l_approx_result = NABOPB(lambda x: bsplinet_test_9d(x), d, basis_flag2, gamma, diminc, s)
C_ssmr1l_approx_result = NABOPB(lambda x: bsplinet_test_9d(x), d, basis_flag3, gamma, diminc, s)

# Show results

if 'C_rand_approx_result' in locals():
    print('====Variant 1====')
    val1 = C_rand_approx_result['val'] * np.sqrt(np.prod(2**(C_rand_approx_result['index'] != 0), axis=1))
    true_val1, real_norm1 = bsplinet_test_9d_chat(C_rand_approx_result['index'])
    err1 = {
        'l2': np.linalg.norm(true_val1 - val1, 2),
        'inf': np.linalg.norm(true_val1 - val1, np.inf)
    }
    print(f'abs. variance: {np.linalg.norm(val1, 2)**2:.5e}')
    print(f'pos. variance: {np.linalg.norm(true_val1, 2)**2:.5e}')
    print(f'true variance: {real_norm1:.5e}')
    print(f'l_2-error: {err1["l2"]:.6e}')
    print(f'max-error: {err1["inf"]:.6e}')
    print(f'samples: {sum(C_rand_approx_result["sample_sizes"])}, candidates: {sum(C_rand_approx_result["cand_sizes"])}, time: {C_rand_approx_result["run_times"][3]:.6f}')

if 'C_mr1l_approx_result' in locals():
    print('====Variant 2====')
    val2 = C_mr1l_approx_result['val'] * np.sqrt(np.prod(2**(C_mr1l_approx_result['index'] != 0), axis=1))
    true_val2, real_norm2 = bsplinet_test_9d_chat(C_mr1l_approx_result['index'])
    err2 = {
        'l2': np.linalg.norm(true_val2 - val2, 2),
        'inf': np.linalg.norm(true_val2 - val2, np.inf)
    }
    print(f'abs. variance: {np.linalg.norm(val2, 2)**2:.5e}')
    print(f'pos. variance: {np.linalg.norm(true_val2, 2)**2:.5e}')
    print(f'true variance: {real_norm2:.5e}')
    print(f'l_2-error: {err2["l2"]:.6e}')
    print(f'max-error: {err2["inf"]:.6e}')
    print(f'samples: {sum(C_mr1l_approx_result["sample_sizes"])}, candidates: {sum(C_mr1l_approx_result["cand_sizes"])}, time: {C_mr1l_approx_result["run_times"][3]:.6f}')

if 'C_ssmr1l_approx_result' in locals():
    print('====Variant 3====')
    val3 = C_ssmr1l_approx_result['val'] * np.sqrt(np.prod(2**(C_ssmr1l_approx_result['index'] != 0), axis=1))
    true_val3, real_norm3 = bsplinet_test_9d_chat(C_ssmr1l_approx_result['index'])
    err3 = {
        'l2': np.linalg.norm(true_val3 - val3, 2),
        'inf': np.linalg.norm(true_val3 - val3, np.inf)
    }
    print(f'abs. variance: {np.linalg.norm(val3, 2)**2:.5e}')
    print(f'pos. variance: {np.linalg.norm(true_val3, 2)**2:.5e}')
    print(f'true variance: {real_norm3:.5e}')
    print(f'l_2-error: {err3["l2"]:e}')
    print(f'max-error: {err3["inf"]:e}')
    print(f'samples: {sum(C_ssmr1l_approx_result["sample_sizes"])}, candidates: {sum(C_ssmr1l_approx_result["cand_sizes"])}, time: {C_ssmr1l_approx_result["run_times"][3]:.6f}')

print('\n')