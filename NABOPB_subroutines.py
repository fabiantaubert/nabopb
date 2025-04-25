import numpy as np
from r1lfft import *
from mr1lfft import *
from cmr1lfft import *
from scipy.fft import dct
from scipy.linalg import lstsq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import LinearOperator

def random_number(d, dims, basis):
    """
    Draw a random anchor `x_tilde` in the remaining dimensions according to the specified basis.

    Parameters
    ----------
    d : int
        Total number of dimensions.
    dims : array_like
        List of fixed dimensions from the set {1, ..., d}.
    basis : string
        Specifies the basis for generating the random anchor. See NABOPB.py 
        for details on the currently supported cases.

    Returns
    -------
    x_tilde : ndarray
        The generated random anchor, an array of shape `(1, d - len(dims))`.

    Raises
    ------
    ValueError
        If the specified `basis` is not recognized or unsupported.
    """

    x_tilde = np.zeros((1, d - dims.size))
    
    if basis in {'Fourier_rand', 'Fourier_r1l', 'Fourier_ssr1l', 'Fourier_mr1l'}:
        x_tilde = np.random.rand(1, d - dims.size)
        
    elif basis in {'Cheby_rand', 'Cheby_mr1l', 'Cheby_mr1l_2024', 'Cheby_mr1l_subsampling'}:
        x_tilde = np.sin(np.pi * (np.random.rand(1, d - dims.size) - 1/2))
        
    else:
        raise ValueError(f'random_number for basis {basis} not defined')
    
    return x_tilde

###############################################################################

def construct_cubature_1d(K, basis, j, d):
    """
    Construct cubature nodes `Xi` and weights `W` for a given one-dimensional candidate set `K`.

    Parameters
    ----------
    K : array_like
        One-dimensional candidate set, typically represented as an array of indices.
    basis : string
        Specifies the basis for generating the cubature nodes and weights. 
        See NABOPB.py for details on the currently supported cases.
    j : int
        Index corresponding to the dimension in consideration.
    d : int
        Total number of dimensions.

    Returns
    -------
    W : xxx
        Cubature weights (or whatever information is needed).
    Xi : ndarray
        Cubature nodes.

    Raises
    ------
    ValueError
        If the specified `basis` is not recognized or unsupported.
    """
    
    W = 0
    Xi = 0
    
    if basis == 'Fourier_rand':
        # Monte Carlo, W is number of nodes
        n = K.shape[0]
        W = int(np.ceil(n * np.log(n)))
        Xi = np.sort(np.random.rand(W))
        
    elif basis in {'Fourier_r1l', 'Fourier_ssr1l', 'Fourier_mr1l'}:
        # Equidistant Points, W is number of nodes
        W = np.max(K) - np.min(K) + 1
        Xi = np.linspace(0, 1 - 1/W, W)
        
    elif basis == 'Cheby_rand':
        # Monte Carlo, W is number of nodes
        n = np.sum(2 ** (np.sum(K != 0, axis=1)))
        W = int(np.ceil(n * np.log(n)))
        Xi = np.sort(np.sin(np.pi * (np.random.rand(W) - 0.5)))
        
    elif basis in {'Cheby_mr1l', 'Cheby_mr1l_subsampling'}:
        # DCT-I of length W+1
        W = np.max(K)
        Xi = np.cos(np.pi * np.linspace(0, 1, W + 1))
        
    else:
        raise ValueError(f'construct_cubature_1d for basis {basis} not defined')
    
    return W, Xi

###############################################################################

def construct_cubature(K, basis, dims, d):
    """
    Construct the cubature nodes `Xi` and weights `W` for a given n-dimensional candidate set `K`.

    Parameters
    ----------
    K : array_like
        n-dimensional candidate set, typically represented as an array of indices.
    basis : string
        Specifies the basis for generating the cubature nodes and weights. 
        See `NABOPB.py` for details on the currently supported cases.
    dims : array_like
        List of fixed dimensions from the set {1, ..., d}.
    d : int
        Total number of dimensions.

    Returns
    -------
    W : int, tuple, or dict
        Reconstruction parameters (e.g. cubature weights).
    Xi : ndarray
        Cubature nodes.

    Raises
    ------
    ValueError
        If the specified `basis` is not recognized or unsupported.
    """

    W = 0
    Xi = 0
    if basis == 'Fourier_rand':
        # Monte Carlo, W is number of nodes
        n = K.shape[0]
        W = int(np.ceil(n * np.log(n)))
        t = K.shape[1]
        Xi = np.random.rand(W, t)
    elif basis == 'Fourier_r1l':
        # Single rank-1 lattice
        z, M = heuristic_lattice_search(K)
        Xi = np.mod(np.arange(M).reshape(-1, 1) * z, M) / M
        W = (z, M)
    elif basis == 'Fourier_ssr1l':
        # Subsampled rank-1 lattice
        z, M = heuristic_lattice_search(K)
        Xi_full = np.mod(np.arange(M).reshape(-1, 1) * z, M) / M
        # Draw n (log n) random indices
        n = K.shape[0]
        N = int(np.ceil(n * np.log(n)))
        idcs_s = np.random.randint(0, M, N)
        # Reduce nodes
        Xi = Xi_full[np.unique(idcs_s), :]
        W = (z, M, idcs_s)
    elif basis == 'Fourier_mr1l':
        # Multiple rank-1 lattice
        arg = {'I': K, 'c': 2, 'C': 1, 'delta': 0.99}
        W = construct_mr1l(arg, 3)
        Ms = W['Ms']
        zs = W['zs']
        Xi = np.zeros((np.sum(Ms) - len(Ms) + 1, K.shape[1]))
        Xi[0:Ms[0], :] = np.mod(np.arange(Ms[0]).reshape(-1, 1) * zs[0], Ms[0]) / Ms[0]
        for j in range(1, len(Ms)):
            Xi[np.sum(Ms[:j]) - j + 1:np.sum(Ms[:j + 1]) - j, :] = np.mod(np.arange(1, Ms[j]).reshape(-1, 1) * zs[j], Ms[j]) / Ms[j]
    elif basis == 'Cheby_rand':
        # Monte Carlo, W is number of nodes
        n = np.sum(2 ** (np.sum(K != 0, axis=1)))
        W = int(np.ceil(n * np.log(n)))
        t = K.shape[1]
        Xi = np.sin(np.pi * (np.random.rand(W, t) - 0.5))
    elif basis == 'Cheby_mr1l':
        # Chebyshev multiple rank-1 lattice
        zs, Ms, reco_infos = multiCR1L_search(K, 4)
        W = {'zs': zs, 'Ms': Ms, 'reco_infos': reco_infos}
        Xi = np.zeros((np.sum(Ms - 1) // 2 + 1, zs.shape[1]))
        Xi[:((Ms[0] + 1) // 2), :] = np.cos(2 * np.pi * np.mod(np.arange((Ms[0] - 1) // 2 + 1).reshape(-1, 1) * zs[0, :], Ms[0]) / Ms[0])
        zaehler = (Ms[0] + 1) // 2
        for j in range(1, len(Ms)):
            Xi[zaehler:zaehler + (Ms[j] - 1) // 2, :] = np.cos(2 * np.pi * np.mod(np.arange(1, (Ms[j] - 1) // 2 + 1).reshape(-1, 1) * zs[j, :], Ms[j]) / Ms[j])
            zaehler += (Ms[j] - 1) // 2
    elif basis == 'Cheby_mr1l_subsampling':
        zs, Ms, reco_infos = multiCR1L_search(K, 4)
        W = {'zs': zs, 'Ms': Ms, 'reco_infos': reco_infos}
        max_os_factor = 3
        W = multi_CR1L_weights(K, W, int(np.ceil(max_os_factor * K.shape[0])))
        Xi = np.cos(2 * np.pi * W['subsampling_data']['xs'])
    else:
        raise ValueError(f'construct_cubature for basis {basis} not defined')

    return W, Xi

###############################################################################

def evaluate_cubature_1d(W, Xi, f, K, basis, j, d):
    """
    Evaluate the reconstructed projected coefficients for a one-dimensional candidate set `K` using
    sampling values `f`, cubature nodes `Xi`, and reconstruction parameters `W`.

    Parameters
    ----------
    W : int, tuple, or dict
        Reconstruction parameters (e.g. cubature weights).
    Xi : array_like
        Cubature nodes.
    f : array_like
        Sampling values for the function to be reconstructed.
    K : array_like
        One-dimensional candidate set, typically represented as an array of indices.
    basis : string
        Specifies the basis for performing the reconstruction. 
        See `NABOPB.py` for details on the currently supported cases.
    j : int
        Index corresponding to the dimension in consideration.
    d : int
        Total number of dimensions.

    Returns
    -------
    f_hat : ndarray
        Reconstructed projected coefficients for the candidate set `K`.

    Raises
    ------
    ValueError
        If the specified `basis` is not recognized or unsupported.
    """

    f_hat = np.zeros((K.shape[0], 1))
    
    if basis == 'Fourier_rand':
        # LSQR
        phi = np.exp(-2 * np.pi * 1j * K @ Xi.T)
        f_hat, _, _, _ = lstsq(np.conjugate(phi.T), f)
    elif basis in ['Fourier_r1l', 'Fourier_ssr1l', 'Fourier_mr1l']:
        # FFT
        # f_hat_temp = np.fft.fftshift(np.fft.fft(f / W, W))
        f_hat_temp = np.roll(np.fft.fft(f / W, W),max(K))
        idx = np.arange(min(K), max(K)+1)
        f_hat = f_hat_temp[np.isin(idx, K)]
    elif basis == 'Cheby_rand':
        # LSQR, check if phi.T is correct
        phi = np.conjugate(np.cos(K @ np.arccos(np.conjugate(Xi.T)))) * (2.0 ** ((K != 0) / 2))
        f_hat, _, _, _ = lstsq(phi.T, f)
    elif basis in ['Cheby_mr1l', 'Cheby_mr1l_subsampling']:
        # DCT-I
        scal = np.ones(W + 1)
        scal[0] = scal[0] / np.sqrt(2)
        scal[W] = scal[W] / np.sqrt(2)
        f_hat_temp = (scal * np.sqrt(2) / np.sqrt(W)) * dct(scal * f, type=1, norm="ortho")
        idx = np.arange(0, max(K) + 1)
        f_hat = f_hat_temp[np.isin(idx, K)]
        
    else:
        raise ValueError(f'evaluate_quadrature_1d for basis {basis} not defined')
    
    return f_hat

###############################################################################

def evaluate_cubature(W, Xi, f, K, basis, dims, d):
    """
    Evaluate the reconstructed projected coefficients for a given n-dimensional candidate set `K`
    using sampling values `f`, cubature nodes `Xi`, and reconstruction parameters `W`.

    Parameters
    ----------
    W : int, tuple, or dict
        Reconstruction parameters (e.g., cubature weights, rank-1 lattice information).
    Xi : array_like
        Cubature nodes, representing the positions used in the reconstruction process.
    f : array_like
        Sampling values of the function to be reconstructed.
    K : array_like
        n-dimensional candidate set, typically represented as an array of indices.
    basis : string
        Specifies the basis used for the reconstruction. 
        See `NABOPB.py` for details on the currently supported cases.
    dims : array_like
        List of fixed dimensions from the set {1, ..., d}.
    d : int
        Total number of dimensions.

    Returns
    -------
    f_hat : ndarray
        Reconstructed projected coefficients for the candidate set `K`.

    Raises
    ------
    ValueError
        If the specified `basis` is not recognized or unsupported.
    """

    f_hat = np.zeros(K.shape[0])
    if basis == 'Fourier_rand':
        # LSQR
        phi = np.exp(-2 * np.pi * 1j * K @ Xi.T)
        f_hat, _, _, _ = lstsq(np.conjugate(phi.T), f)
    elif basis == 'Fourier_r1l':
        # Rank-1 lattice FFT
        f_hat = lfft(K, W[0], W[1], f, 'transp') / W[1]
    elif basis == 'Fourier_ssr1l':
        # Subsampled Rank-1 lattice FFT
        # Recover f
        _, _, ic = np.unique(W[2], return_index=True, return_inverse=True)
        f_full = f[ic]
        # Build sparse matrix
        S = csr_matrix((np.ones(len(W[2])), (W[2], np.arange(len(W[2])))), shape=(W[1], len(W[2])))
        # LFFT
        A = LinearOperator((len(f_full), len(f_hat)), matvec=lambda x: Ax_ssr1l(x, K, W, S), rmatvec=lambda b: Atb_ssr1l(b, K, W, S))
        f_hat = lsqr(A, f_full)[0]
        f_hat = f_hat / np.sqrt(len(W[2]))
    elif basis == 'Fourier_mr1l':
        # Multiple rank-1 lattice FFT
        opts = {'alg': 'cg'}
        err = 10**-12
        f_hat = multi_lattice_ifft(f, K, W, err, opts)[0]
    elif basis == 'Cheby_rand':
        # LSQR
        Xi = np.arccos(Xi)
        phi = np.zeros((K.shape[0], f.shape[0]))
        for i in range(K.shape[0]):
            phi[i, :] = np.prod(np.cos(np.dot(np.ones((Xi.shape[0], 1)), K[i, :].reshape(1, -1)) * Xi), axis=1) * (2.0 ** (np.sum(K[i,:] != 0) / 2))
        f_hat, _, _, _ = lstsq(phi.T, f)
    elif basis == 'Cheby_mr1l':
        # Chebyshev multiple rank-1 lattice FFT
        A = LinearOperator((len(f), len(f_hat)), matvec=lambda x: Ax_cmr1l(x, K, W), rmatvec=lambda b: Atb_cmr1l(b, K, W))
        f_hat = lsqr(A, f)[0]
    elif basis == 'Cheby_mr1l_subsampling':
        A = LinearOperator((len(f), len(f_hat)), matvec=lambda x: Ax_cmr1lss(x, K, W), rmatvec=lambda b: Atb_cmr1lss(b, K, W))
        f_hat = lsqr(A, f)[0]
    else:
        raise ValueError(f'evaluate_quadrature for basis {basis} not defined')
    
    return f_hat     

def Ax_ssr1l(x, K, W, S):
    return S.T @ lfft(K, W[0], W[1], x, 'notransp') / np.sqrt(len(W[2]))

def Atb_ssr1l(b, K, W, S):
    return lfft(K, W[0], W[1], S @ b, 'transp') / np.sqrt(len(W[2]))

def Ax_cmr1l(x, K, W):
    return multi_CR1L_FFT(x, K, W, 'notransp')

def Atb_cmr1l(b, K, W):
    return multi_CR1L_FFT(b, K, W, 'transp')

def Ax_cmr1lss(x, K, W):
    return multi_CR1L_subsampled_FFT(x, K, W, 'notransp')

def Atb_cmr1lss(b, K, W):
    return multi_CR1L_subsampled_FFT(b, K, W, 'transp')