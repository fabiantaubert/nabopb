import numpy as np
from r1lfft import *

def multi_lattice_fft(fhat, h, lattice, tflag='notransp'):
    """
    Compute the multi-lattice FFT (Fast Fourier Transform) or its transpose over a multiple rank-1 lattice.

    Parameters
    ----------
    fhat : array_like
        Fourier coefficients or function values. If `tflag` is 'notransp', 
        `fhat` represents the Fourier coefficients f_hat(j). If `tflag` is 'transp', 
        `fhat` represents the function values evaluated at each lattice point.
    h : array_like
        Frequency index set, represented as an array of integers.
    lattice : dict
        Dictionary defining the multiple rank-1 lattice with the following keys: 
        - 'Ms' (array_like): List or array of lattice sizes.
        - 'zs' (array_like): List or array of generating integer vectors.
    tflag : {'notransp', 'transp', 'bothreal', 'both'}, optional
        Specifies the type of FFT operation to perform.
        
        - 'notransp': Computes the forward FFT across lattice segments.
        
        - 'transp': Computes the transposed (inverse) FFT across lattice segments.
        
        - 'bothreal': Applies both forward and inverse transforms, returning only 
          the real part of the result.
        
        - 'both': Applies both forward and inverse transforms, returning the complex result.
          
        Default is 'notransp'.

    Returns
    -------
    array_like
        The result of the FFT operation, based on the specified `tflag`. If `tflag` is 'bothreal', 
        the result is real-valued; otherwise, it may be complex.
    """
    
    Ms = lattice['Ms']
    zs = lattice['zs']

    if tflag == 'notransp':
        y = multi_lattice_fft_notransp(fhat, h, Ms, zs)
    elif tflag == 'transp':
        y = multi_lattice_fft_transp(fhat, h, Ms, zs)
    elif tflag == 'bothreal':
        y = np.real(multi_lattice_fft_transp(multi_lattice_fft_notransp(fhat, h, Ms, zs), h, Ms, zs))
    elif tflag == 'both':
        y = multi_lattice_fft_transp(multi_lattice_fft_notransp(fhat, h, Ms, zs), h, Ms, zs)

    return y

def multi_lattice_fft_notransp(fhat, h, Ms, zs):
    a = []
    for j in range(len(Ms)):
        a1 = lfft(h, zs[j], Ms[j], fhat, 'notransp')
        istart = 0 if j == 0 else 1
        a.extend(a1[istart:])
    return np.array(a)

def multi_lattice_fft_transp(fhat, h, Ms, zs):
    a = np.zeros(h.shape[0])
    for j in range(len(Ms)):
        if j == 0:
            f = fhat[:Ms[0]]
        else:
            f = np.concatenate(([0], fhat[sum(Ms[:j]) - j + 1:sum(Ms[:j+1]) - j]))
        a = a + lfft(h, zs[j], Ms[j], f, 'transp')
    return a

