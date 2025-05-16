import numpy as np
import time
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
import os
import sys
import inspect

# Set up directory paths to import CMR1LFFT
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from cmr1lfft import *

def solve_pois1d(var):
    # Split input array: x (spatial variable) and a (Fourier coefficients)
    x = var[:, 0]
    a = var[:, 1:]

    # Determine the index range for Fourier coefficients
    N = (a.shape[1] - 1) / 2
    if not N.is_integer():
        raise ValueError('N is not an integer.')
    N = int(N)

    ind = np.linspace(-N, N, 2 * N + 1)  # Index range from -N to N

    # Transform spatial variable x from [-1, 1] to [0, 1]
    x = (x + 1) / 2

    # Compute the solution u(x, a) using the exact formula
    with np.errstate(divide='ignore', invalid='ignore'):
        temp = a / (4 * np.pi**2 * ind**2) * (np.exp(2 * np.pi * 1j * np.expand_dims(ind, 0) * np.expand_dims(x, 1)) - 1)
    val = a[:, N] / 2 * x * (1 - x) + np.nansum(temp, axis=1)

    return val

def eval_cheby(x, val, index):
    # Evaluate the Chebyshev polynomial
    res = np.zeros((x.shape[0]), dtype=np.complex_)
    scal = (2.0 ** (np.sum(index != 0, axis=1) / 2))
    for j in range(x.shape[0]):
        res[j] = np.sum(val * scal * np.prod(np.cos(np.arccos(x[j, :]) * index), axis=1))
    return res

# Functions to define the Linear Operator for LSQR
def Ax_cmr1l(x, K, W):
    return multi_CR1L_FFT(x, K, W, 'notransp')

def Atb_cmr1l(b, K, W):
    return multi_CR1L_FFT(b, K, W, 'transp')

# Set index set parameters
dim = 100
x_ind_max = 999

# Build the index set.
t1 = time.time()
dim1 = np.arange(x_ind_max + 1)

N = len(dim1)
rows = np.zeros((N * dim, dim), dtype=int)
rows[:, 0] = np.repeat(dim1, dim)
eye_block = np.tile(np.vstack(([0]*(dim-1), np.eye(dim - 1, dtype=int))), (N, 1))
rows[:, 1:] = eye_block

idx_set = rows
time_1 = time.time() - t1

# Build the CMR1L
t2 = time.time()
zs, Ms, reco_infos = multiCR1L_search(idx_set, 4)
W = {'zs': zs, 'Ms': Ms, 'reco_infos': reco_infos}
Xi = np.zeros((np.sum(Ms - 1) // 2 + 1, zs.shape[1]))
Xi[:((Ms[0] + 1) // 2), :] = np.cos(2 * np.pi * np.mod(np.arange((Ms[0] - 1) // 2 + 1).reshape(-1, 1) * zs[0, :], Ms[0]) / Ms[0])
zaehler = (Ms[0] + 1) // 2
for j in range(1, len(Ms)):
    Xi[zaehler:zaehler + (Ms[j] - 1) // 2, :] = np.cos(2 * np.pi * np.mod(np.arange(1, (Ms[j] - 1) // 2 + 1).reshape(-1, 1) * zs[j, :], Ms[j]) / Ms[j])
    zaehler += (Ms[j] - 1) // 2
time_2 = time.time() - t2

# Sampling by solving the Poisson equation
t3 = time.time()
f = solve_pois1d(Xi)
time_3 = time.time() - t3

# Evaluating the CMR1L / Computing the coefficients f_hat
t4 = time.time()
f_hat = np.zeros(idx_set.shape[0])
A = LinearOperator((len(f), len(f_hat)), matvec=lambda x: Ax_cmr1l(x, idx_set, W), rmatvec=lambda b: Atb_cmr1l(b, idx_set, W))
maxit = 100
tol = 1e-8
f_hat = lsqr(A, f, atol = tol, btol = tol, iter_lim = maxit)[0]
time_4 = time.time() - t4

# Testing and Plotting for a single function f
a = (np.random.rand(1, dim - 1) * 2 - 1) * 1
a = np.tile(a, (1000, 1))

# Evaluate the approximation
x = np.linspace(-1, 1, 1000).reshape(-1, 1)
t5 = time.time()
y2 = eval_cheby(np.hstack([x, a]), f_hat, idx_set)
time_5 = time.time() - t5

# Compute the exact solution
t6 = time.time()
y1 = solve_pois1d(np.hstack([x, a]))
time_6 = time.time() - t6

# Plot both solutions
plt.figure(1)
plt.plot(x, np.real(y1))
plt.plot(x, np.real(y2))
plt.figure(2)
plt.plot(x, np.imag(y1))
plt.plot(x, np.imag(y2))

# Compute the relative l2-error
err = np.linalg.norm(y1 - y2, 2) / np.linalg.norm(y1, 2)

# Console Output
print(f'The relative error is {err:e}')
print(f'Total time: {time_1 + time_2 + time_3 + time_4 + time_5 + time_6} seconds. Times in detail:')
print(f'Step 1 - Build the index set:     {time_1} seconds')
print(f'Step 2 - Build the CMR1L:         {time_2} seconds')
print(f'Step 3 - Sample the solution:     {time_3} seconds')
print(f'Step 4 - Compute coefficients:    {time_4} seconds')
print(f'Step 5 - Evaluate Cheby. poly.:   {time_5} seconds')
print(f'Step 6 - Compute exact solution:  {time_6} seconds')

# Computing the l2-error
N = 10**2  # Amount of random draws of a
a = (np.random.rand(N, dim - 1) * 2 - 1) * 1
x = np.linspace(-1, 1, 1000).reshape(-1, 1)

err = np.zeros(N)
for i in range(N):
    a_temp = np.tile(a[i, :], (1000, 1))
    # Exact solution
    y1 = solve_pois1d(np.hstack([x, a_temp]))
    # Solution by our approximation
    y2 = eval_cheby(np.hstack([x, a_temp]), f_hat, idx_set)
    # Relative l2-error
    err[i] = np.linalg.norm(y1 - y2, 2) / np.linalg.norm(y1, 2)
    
plt.boxplot(err, vert=False)
plt.xscale('log')
plt.show()
