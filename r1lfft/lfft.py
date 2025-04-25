import numpy as np

def lfft(h, z, M, f_hat, tflag):
    """
    Compute the lattice FFT (Fast Fourier Transform) or its transpose.

    Parameters
    ----------
    h : array_like
        Frequency index set, represented as an array of integers.
    z : array_like
        Generating integer vector.
    M : int
        Lattice size.
    fhat : array_like
        Fourier coefficients or function values. If `tflag` is 'notransp', 
        `fhat` represents the Fourier coefficients f_hat(j). If `tflag` is 'transp', 
        `fhat` represents the function values evaluated at mod(j*z/M, 1).
    tflag : {'notransp', 'transp'}
        Specifies whether to evaluate the forward FFT or its transpose.
        
        - 'notransp': Computes the FFT as:
          f(j+1) = sum_{k in h} f_hat(k) * exp(-2*pi*i*j*k^T*z/M) for j=0,...,M-1
        
        - 'transp': Computes the transposed FFT as:
          f(l) = sum_{j=0}^M f_hat(j) * exp(2*pi*i*j*k(l)^T*z/M) for l=1,...,size(h,1),
          where k(l) = h(l,:)

    Returns
    -------
    array_like
        The computed FFT or transposed FFT values, depending on the value of `tflag`.

    Notes
    -----
    When `z` and `M` generate a reconstruction lattice, the following identity holds:
    lfft(h, z, M, lfft(h, z, M, fhat, 'notransp'), 'transp') = M * fhat
    """
    
    if not np.issubdtype(h.dtype, np.floating):
        h = h.astype(float) 
    if tflag == 'notransp':
        f = lhcfft_notransp(h, z, M, f_hat)
    elif tflag == 'transp':
        f = lhcfft_transp(h, z, M, f_hat)
    return f

def lhcfft_notransp(H, z, M, f_hat):
    k = (H @ z) % M + 1  # Using matrix multiplication and modulo
    if np.iinfo(np.int64).max < np.max(k):
        print('Warning: int64 problem')
    k = k.astype(np.int64)
    fhat1 = np.zeros(M, dtype=np.complex_)
    fhat1[:np.max(k)] = np.bincount(k-1, weights = np.real(f_hat)) + 1j * np.bincount(k-1, weights = np.imag(f_hat)) # Accumulate values at indices k-1 
    f = M * np.fft.ifft(fhat1)
    return f

def lhcfft_transp(H, z, M, f):
    k = (H @ z) % M + 1
    if np.iinfo(np.int64).max < np.max(k):
        print('Warning: int64 problem')
    k = k.astype(np.int64)
    ghat = np.fft.fft(f)
    fhat = ghat[k - 1]  # Adjust for zero-based indexing in Python   
    return fhat
