import numpy as np
from scipy.fft import fft, ifft

def multi_CR1L_FFT(phat, I, multi_lattice, tflag='notransp'):
    """
    Performs a multi-dimensional FFT computation for multiple CR1L.

    Parameters
    ----------
    phat : ndarray
        The input coefficients or sampling values depending on the value of `tflag`:
        - For `tflag='notransp'`, `phat` represents the Chebyshev coefficients.
        - For `tflag='transp'`, `phat` represents the sampling values.
    I : ndarray
        Index set of Chebyshev polynomials, a matrix of shape (|I|, d), where |I| is the number of indices
        and `d` is the spatial dimension. Each row corresponds to a multi-index.
    multi_lattice : dict
        Dictionary containing information about the multi-dimensional CR1L:
        - `Ms` (ndarray): Lattice sizes.
        - `zs` (ndarray): Generating vectors.
        - `reco_infos` (dict): Reconstruction metadata, including:
            - `M_I_index_I` (ndarray): The mirrored index mapping.
            - `kz_mod_M` (list of ndarrays): The modulo results for each lattice.
            - `cur_j` (int): Current dimension being processed.
    tflag : {'notransp', 'transp', 'notransp-transp'}, optional
        Indicates the type of operation to perform (default: 'notransp'):
        - `'notransp'`: Compute sampling values using the FFT.
        - `'transp'`: Compute the transpose operation, recovering coefficients.
        - `'notransp-transp'`: Compute a specialized dual operation for reconstruction.

    Returns
    -------
    p : ndarray
        The computed FFT output, either the sampling values or reconstructed coefficients,
        depending on the `tflag`.

    Raises
    ------
    ValueError
        If `tflag` is not one of {'notransp', 'transp', 'notransp-transp'}.
    """
    Ms = multi_lattice['Ms']
    zs = multi_lattice['zs']
    reco_infos = multi_lattice['reco_infos']

    if tflag == 'notransp':
        p = np.zeros((sum(Ms) - len(Ms)) // 2 + 1, dtype=np.complex64)
        reco_infos['cur_j'] = 1
        p[:(Ms[0] + 1) // 2] = single_CR1L_FFT_new(phat, I, zs[0], Ms[0], tflag, reco_infos)
        zaehler = (Ms[0] + 1) // 2
        for j in range(1, len(Ms)):
            reco_infos['cur_j'] = j + 1
            tmp = single_CR1L_FFT_new(phat, I, zs[j], Ms[j], tflag, reco_infos)
            p[zaehler:zaehler + (Ms[j] - 1) // 2] = tmp[1:]
            zaehler += (Ms[j] - 1) // 2

    elif tflag == 'transp':
        p = np.zeros(I.shape[0], dtype=np.complex64)
        reco_infos['cur_j'] = 1
        p = single_CR1L_FFT_new(phat[:(Ms[0] + 1) // 2], I, zs[0], Ms[0], tflag, reco_infos)
        zaehler = (Ms[0] + 1) // 2
        for j in range(1, len(Ms)):
            reco_infos['cur_j'] = j + 1
            tmp = np.zeros(1, dtype=np.complex64)
            tmp = np.append(tmp, phat[zaehler:zaehler + (Ms[j] - 1) // 2])
            zaehler += (Ms[j] - 1) // 2
            p += single_CR1L_FFT_new(tmp, I, zs[j], Ms[j], tflag, reco_infos)

    elif tflag == 'notransp-transp':
        p = np.zeros_like(phat, dtype=np.complex64)
        for t in range(phat.shape[1]):
            ptmp = np.zeros((sum(Ms) - len(Ms)) // 2 + 1, dtype=np.complex64)
            phattmp = phat[:, t]
            fhat = np.zeros(len(reco_infos['M_I_index_I']), dtype=np.complex64)
            fhat = phattmp[reco_infos['M_I_index_I']] / (2.0 ** (np.sum(I[reco_infos['M_I_index_I']] != 0, axis=1) / 2))
            fhat1 = np.bincount(reco_infos['kz_mod_M'][0], weights=fhat, minlength=Ms[0])
            f = Ms[0] * ifft(fhat1)
            ptmp[:(Ms[0] + 1) // 2] = f[:(Ms[0] + 1) // 2]

            zaehler = (Ms[0] + 1) // 2
            for j in range(1, len(Ms)):
                fhat = np.zeros(len(reco_infos['M_I_index_I']), dtype=np.complex64)
                fhat = phattmp[reco_infos['M_I_index_I']] / (2.0 ** (np.sum(I[reco_infos['M_I_index_I']] != 0, axis=1) / 2))
                fhat1 = np.bincount(reco_infos['kz_mod_M'][j], weights=fhat, minlength=Ms[j])
                f = Ms[j] * ifft(fhat1)
                ptmp[zaehler:zaehler + (Ms[j] - 1) // 2] = f[1:(Ms[j] + 1) // 2]
                zaehler += (Ms[j] - 1) // 2

            ptmp1 = np.zeros(Ms[0], dtype=np.complex64)
            ptmp1[:(Ms[0] + 1) // 2] = ptmp[:(Ms[0] + 1) // 2]
            ghat = fft(ptmp1)
            fhat = ghat[reco_infos['kz_mod_M'][0]]
            p[:, t] = np.bincount(reco_infos['M_I_index_I'], weights=fhat) / (2.0 ** (np.sum(I != 0, axis=1) / 2))

            zaehler = (Ms[0] + 1) // 2
            for j in range(1, len(Ms)):
                ptmp1 = np.zeros(Ms[j], dtype=np.complex64)
                ptmp1[1:(Ms[j] + 1) // 2] = ptmp[zaehler:zaehler + (Ms[j] - 1) // 2]
                zaehler += (Ms[j] - 1) // 2
                ghat = fft(ptmp1)
                fhat = ghat[reco_infos['kz_mod_M'][j]]
                p[:, t] += np.bincount(reco_infos['M_I_index_I'], weights=fhat) / (2.0 ** (np.sum(I != 0, axis=1) / 2))

    else:
        raise ValueError('unknown tflag')

    return p

def single_CR1L_FFT_new(phat, I, z, M, tflag, reco_infos):
    if tflag == 'notransp':
        M_I_index_I = reco_infos['M_I_index_I']
        k = reco_infos['kz_mod_M'][reco_infos['cur_j'] - 1]

        fhat = np.zeros(len(M_I_index_I), dtype=np.complex64)
        fhat = phat[M_I_index_I-1] / (2.0 ** (np.sum(I[M_I_index_I-1] != 0, axis=1) / 2))
        fhat1 = np.bincount(k, weights=np.real(fhat), minlength=M) + 1j * np.bincount(k, weights=np.imag(fhat), minlength=M)
        f = M * ifft(fhat1)
        p = f[:(M + 1) // 2]

    elif tflag == 'transp':
        M_I_index_I = reco_infos['M_I_index_I']
        k = reco_infos['kz_mod_M'][reco_infos['cur_j'] - 1]
        phat = np.pad(phat, (0, len(phat) - 1))
        ghat = fft(phat)
        fhat = ghat[k]
        p = (np.bincount(M_I_index_I-1, weights=np.real(fhat)) + 1j * np.bincount(M_I_index_I-1, weights=np.imag(fhat))) / (2.0 ** (np.sum(I != 0, axis=1) / 2))
        
    else:
        raise ValueError('unknown tflag')

    return p

