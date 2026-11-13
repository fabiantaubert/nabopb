import numpy as np
from scr1lfft.SCR1L_FFT import hit_numbers, orbit_sum, _get_reco_infos


def SCR1L_weights(I, lattice, wflag="weighted"):
    """
    Row and column weights of the weighted least squares problem on a shifted cosine rank-1 lattice
    (Algorithm 6.1, step 2, of D. Potts and T. Ullrich, Cosine approximation on shifted rank-1 lattices).

    The least squares problem is solved in the frequency domain,
        min_y || W (B(t) V y - ghat) ||_2,  c = V y,  ghat = fft(g) / M,
    with the coefficients c in the cosine basis phi_h(x) = prod_j cos(2 pi h_j x_j).

    Parameters
    ----------
    I : ndarray
        Index set of shape (|I|, d) with nonnegative entries.
    lattice : dict
        Dictionary with `z`, `M`, `delta` and, optionally, `reco_infos`.
    wflag : {'weighted', 'unweighted'}, optional
        - `'weighted'` (default): w_l = H(l)^{-1/2} for the hit numbers H(l) > 0 and w_l = 0 otherwise,
          and v_h = omega_h^{-1/2} with omega_h = 2^{-2 s(h)} sum_{eps in E(h)} w_{sigma(h, eps)}^2, so
          that the expectation of the Gram matrix over a random shift is the identity.
        - `'unweighted'`: w_l = 1 for H(l) > 0 and v_h = 2^{s(h)/2}, i.e., ordinary least squares on
          all samples in the orthonormal Chebyshev basis.

    Returns
    -------
    weights : dict
        - `w` (ndarray): row weights on Z_M.
        - `v` (ndarray): column weights on I.
        - `H` (ndarray): hit numbers on Z_M.
    """
    reco_infos = _get_reco_infos(I, lattice)
    M = int(lattice["M"])
    H = hit_numbers(reco_infos, M, I.shape[0])
    s = reco_infos["nnz_s"]
    if wflag == "weighted":
        w = np.zeros(M)
        w[H > 0] = 1 / np.sqrt(H[H > 0])
        omega = orbit_sum(reco_infos, w**2) / 4.0**s
        v = 1 / np.sqrt(omega)
    elif wflag == "unweighted":
        w = (H > 0).astype(float)
        v = 2.0 ** (s / 2)
    else:
        raise ValueError("unknown wflag")
    return {"w": w, "v": v, "H": H}
