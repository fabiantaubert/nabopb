import numpy as np
from mr1lfft import *
import time

def multi_lattice_ifft(f, h, lattice, err=1e-10, opts=None):
    """
    Compute the inverse multi-lattice FFT (iFFT) over a multiple rank-1 lattice.

    Parameters
    ----------
    f : array_like
        Values at the nodes of the multiple rank-1 lattice.
    h : array_like
        Frequency index set, represented as an array of integers.
    lattice : dict
        Dictionary defining the multiple rank-1 lattice with the following keys: 
        - 'Ms' (array_like): List or array of lattice sizes.
        - 'zs' (array_like): List or array of generating integer vectors.
    err : float, optional
        Tolerated error for the conjugate gradient (CG) algorithm. Default is 1e-10.
    opts : dict, optional
        Additional optional parameters:
        
        - 'alg' (str): Algorithm to use. Options include:
            - 'cg': Use conjugate gradient (default).
            - 'direct': Use direct computation (requires a specific lattice construction).
            - 'direct2', 'direct3', 'consecutive': Variants of direct computation.
        - 'start' (array_like): Initial guess for the CG algorithm.
        - 'maxit' (int): Maximum number of iterations for the CG algorithm.

    Returns
    -------
    fhat : array_like
        The computed inverse FFT result.
    time : float
        The time taken to compute the result.
    numiter : int
        The number of iterations performed (relevant for 'cg' and 'consecutive' algorithms).

    Raises
    ------
    ValueError
        If the chosen algorithm is incompatible with the lattice structure or is invalid.
    """
    
    Ms = lattice['Ms']
    zs = lattice['zs']

    if opts is None:
        opts = {'alg': 'cg'}
    if 'maxit' not in opts:
        opts['maxit'] = float('inf')

    if opts['alg'] == 'cg':
        fhat, computation_time, numiter = ml_ifft_cg(f, h, lattice, err, opts)
    elif opts['alg'] == 'direct':
        if lattice.get('mark', False):
            h = h.astype(float)
            fhat, computation_time = ml_ifft_direct(f, h, Ms, zs)
            numiter = 1
        else:
            raise ValueError('For direct ifft methods the lattice may not be constructed via algorithm 3-5 in construct_mr1l.m')
    elif opts['alg'] == 'direct2':
        if lattice.get('mark', False):
            fhat, computation_time = ml_ifft_direct2(f, h, Ms, zs)
            numiter = 1
        else:
            raise ValueError('For direct ifft methods the lattice may not be constructed via algorithm 3-5 in construct_mr1l.m')
    elif opts['alg'] == 'direct3':
        if lattice.get('mark', False):
            fhat, computation_time = ml_ifft_direct3(f, h, Ms, zs)
            numiter = 1
        else:
            raise ValueError('For direct ifft methods the lattice may not be constructed via algorithm 3-5 in construct_mr1l.m')
    elif opts['alg'] == 'consecutive':
        if lattice.get('mark', False):
            fhat, time1 = ml_ifft_direct(f, h, Ms, zs)
            opts['start'] = fhat
            fhat, time2, numiter = ml_ifft_cg(f, h, lattice, err, opts)
            computation_time = time1 + time2
        else:
            raise ValueError('For the consecutive ifft method the lattice may not be constructed via algorithm 3-5 in construct_mr1l.m')
    else:
        raise ValueError(f"Unknown algorithm: {opts['alg']}")

    return fhat, computation_time, numiter

def ml_ifft_cg(f, h, lattice, err, opts):
    start_time = time.time()
    f1 = multi_lattice_fft(f, h, lattice, 'transp')

    if 'start' not in opts:
        opts['start'] = np.zeros(h.shape[0])

    fhat, numiter = dirtycg(lambda x: multi_lattice_fft(x, h, lattice, 'both'), f1, err, opts)
    computation_time = time.time() - start_time

    return fhat, computation_time, numiter

def ml_ifft_direct(f, h, Ms, zs):
    start_time = time.time()
    fhat = np.zeros(h.shape[0])
    restind = np.arange(h.shape[0])

    for j in range(len(Ms)):
        if j == 0:
            f1 = f[:Ms[0]]
            a1 = 1/Ms[0] * lfft(h, zs[j], Ms[j], f1, 'transp')
        else:
            f1 = np.concatenate(([f[0]], f[sum(Ms[:j])-j+2:sum(Ms[:j+1])-j]))
            a1 = 1/Ms[j] * lfft(h, zs[j], Ms[j], f1 - lfft(h, zs[j], Ms[j], fhat, 'notransp'), 'transp')

        y = np.mod(h[restind] @ zs[j], Ms[j])
        ind = np.argsort(y)
        ind2 = np.setdiff1d(np.arange(len(y)), np.where(np.diff(y[ind]) == 0)[0] + 1)
        fhat[restind[ind[ind2]]] = a1[restind[ind[ind2]]]
        restind = np.setdiff1d(restind, restind[ind[ind2]])

    computation_time = time.time() - start_time
    return fhat, computation_time

def ml_ifft_direct2(f, h, Ms, zs):
    start_time = time.time()
    fhat = np.zeros(h.shape[0])
    recoind = np.array([], dtype=int)
    restind = np.arange(h.shape[0])

    for j in range(len(Ms)):
        y = np.mod(h[restind] @ zs[j], Ms[j])
        ind = np.argsort(y)
        ind2 = np.setdiff1d(np.arange(len(y)), np.where(np.diff(y[ind]) == 0)[0] + 1)
        aktind = restind[ind[ind2]]

        if j == 0:
            f1 = f[:Ms[0]]
            a1 = 1/Ms[0] * lfft(h[aktind], zs[j], Ms[j], f1, 'transp')
        else:
            f1 = np.concatenate(([f[0]], f[sum(Ms[:j])-j+2:sum(Ms[:j+1])-j]))
            a1 = 1/Ms[j] * lfft(h[aktind], zs[j], Ms[j], f1 - lfft(h[recoind], zs[j], Ms[j], fhat[recoind], 'notransp'), 'transp')

        fhat[aktind] = a1
        restind = np.setdiff1d(restind, aktind)
        recoind = np.union1d(recoind, aktind)

    computation_time = time.time() - start_time
    return fhat, computation_time

def ml_ifft_direct3(f, h, Ms, zs):
    start_time = time.time()
    fhat = np.zeros(h.shape[0])
    recoind = np.array([], dtype=int)
    restind = np.arange(h.shape[0])

    for j in range(len(Ms)):
        y = np.mod(h[restind] @ zs[j], Ms[j])
        ind = np.argsort(y)
        ind2 = np.setdiff1d(np.arange(len(y)), np.where(np.diff(y[ind]) == 0)[0] + 1)
        aktind = restind[ind[ind2]]

        if j == 0:
            f1 = f[:Ms[0]]
            a1 = 1/Ms[0] * lfft(h[aktind], zs[j], Ms[j], f1, 'transp')
        else:
            f1 = np.concatenate(([f[0]], f[sum(Ms[:j])-j+2:sum(Ms[:j+1])-j]))
            inds = np.mod(h[recoind] @ zs[j], Ms[j])
            asubs = np.bincount(inds, weights=fhat[recoind], minlength=Ms[j])
            a1 = 1/Ms[j] * lfft(h[aktind], zs[j], Ms[j], f1, 'transp') - asubs[np.mod(h[aktind] @ zs[j], Ms[j])]

        fhat[aktind] = a1
        restind = np.setdiff1d(restind, aktind)
        recoind = np.union1d(recoind, aktind)

    computation_time = time.time() - start_time
    return fhat, computation_time

def dirtycg(fhandle, b, err, opts):
    if 'start' not in opts:
        opts['start'] = np.zeros_like(b)

    err2 = err * err
    x = opts['start']
    r = b if np.linalg.norm(x) == 0 else b - fhandle(x)
    p = r
    rsold = np.conjugate(r.T) @ r

    if rsold != 0:
        for j in range(1, len(b) + 1):
            Ap = fhandle(p)
            alpha = rsold / (np.conjugate(p.T) @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.conjugate(r.T) @ r
            if rsnew < err2:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            if j == opts['maxit']:
                break
    else:
        j = 0

    return x, j
