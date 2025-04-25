import numpy as np
from cmr1lfft import *

def multi_CR1L_subsampled_FFT(phat, I, multi_lattice, tflag='notransp', wflag='unweighted'):
    """
    Computes the subsampled FFT-based evaluation or reconstruction of polynomials using multiple CR1L.

    Parameters
    ----------
    phat : ndarray
        Chebyshev coefficients or sampling values depending on the `tflag`:
        - For `tflag='notransp'`: Input is the Chebyshev coefficients.
        - For `tflag='transp'`: Input is the sampling values.
    I : ndarray
        Index set of Chebyshev polynomials, a matrix of shape (|I|, d), where |I| is the number of indices
        and `d` is the spatial dimension. Each row corresponds to a multi-index.
    multi_lattice : dict
        Contains data related to the multi-dimensional CR1L lattice and subsampling:
        - `subsampling_data['js']`: Indices for subsampling.
        - `subsampling_data['ws']`: Precomputed weights for subsampling.
        - `Ms`: Lattice sizes.
    tflag : {'notransp', 'transp'}, optional
        Algorithm selection flag (default: 'notransp'):
        - `'notransp'`: Computes sampling values of the polynomial at the (subsampled) nodes.
            - If `wflag='unweighted'`, computes `C * phat`.
            - If `wflag='weighted'`, computes `W^{1/2} * C * phat`.
        - `'transp'`: Computes the transpose of the `'notransp'` operation:
            - If `wflag='unweighted'`, computes `C^T * phat`.
            - If `wflag='weighted'`, computes `C^T * W^{1/2} * phat`.
    wflag : {'unweighted', 'weighted'}, optional
        Weight flag for computation (default: 'unweighted'):
        - `'unweighted'`: Ignores weights in the computation.
        - `'weighted'`: Applies precomputed weights to stabilize the reconstruction.

    Returns
    -------
    p : ndarray
        The resulting array after applying the subsampled FFT or its transpose, depending on the `tflag` and `wflag`.

    Raises
    ------
    ValueError
        If `tflag` is not one of {'notransp', 'transp'}.
        If `wflag` is not one of {'unweighted', 'weighted'}.
    """

    if wflag not in ['weighted', 'unweighted']:
        raise ValueError('unknown wflag')

    if tflag == 'notransp':
        p_notsubsampled = multi_CR1L_FFT(phat, I, multi_lattice, 'notransp')
        p = p_notsubsampled[multi_lattice['subsampling_data']['js']]
        
        # apply weight matrix if requested
        if wflag == 'weighted':
            p = np.sqrt(multi_lattice['subsampling_data']['ws']) * p

    elif tflag == 'transp':
        # apply weight matrix if requested
        if wflag == 'weighted':
            phat = np.sqrt(multi_lattice['subsampling_data']['ws']) * phat

        # pigeon-hole sampling values
        p = np.zeros(np.sum((multi_lattice['Ms'] - 1) // 2) + 1)
        p[multi_lattice['subsampling_data']['js']] = phat
        
        # apply transposed FFT
        p = multi_CR1L_FFT(p, I, multi_lattice, 'transp')

    else:
        raise ValueError('unknown tflag')

    return p

