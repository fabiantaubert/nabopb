import numpy as np
from scipy.fft import fft
from scipy.sparse.linalg import lsqr
from scr1lfft.build_aliasing_structure import build_aliasing_structure
from scr1lfft.SCR1L_FFT import SCR1L_FFT, SCR1L_nodes
from scr1lfft.SCR1L_weights import SCR1L_weights
from scr1lfft.SCR1L_reconstruct import _FreqLS


def multi_SCR1L_FFT(phat, I, multi_lattice, tflag="notransp"):
    """
    Fast evaluation of Chebyshev polynomials on a union of shifted cosine rank-1 lattices and its
    transpose.

    Parameters
    ----------
    phat : ndarray
        The input coefficients or sampling values depending on the value of `tflag`:
        - For `tflag='notransp'` and `tflag='notransp-transp'`, `phat` contains the coefficients in the
          orthonormal Chebyshev basis, shape (|I|,) or (|I|, r).
        - For `tflag='transp'`, `phat` contains the sampling values of all lattices, stacked in the order
          of the lattices, shape (sum(Ms),) or (sum(Ms), r).
    I : ndarray
        Index set of Chebyshev polynomials, shape (|I|, d).
    multi_lattice : dict
        Dictionary containing the union of shifted lattices:
        - `zs` (ndarray): Generating vectors, shape (R, d).
        - `Ms` (ndarray): Lattice sizes.
        - `deltas` (ndarray): Shifts, shape (R, d).
        - `reco_infos` (list of dict, optional): Aliasing structures, computed if not provided.
    tflag : {'notransp', 'transp', 'notransp-transp'}, optional
        Indicates the type of operation to perform (default: 'notransp'):
        - `'notransp'`: Compute the sampling values on all lattices.
        - `'transp'`: Compute the transpose.
        - `'notransp-transp'`: Compute A^T A phat without FFT.

    Returns
    -------
    p : ndarray
        The real sampling values or coefficients, depending on `tflag`.

    Raises
    ------
    ValueError
        If `tflag` is not one of {'notransp', 'transp', 'notransp-transp'}.
    """
    lattices = _lattices(I, multi_lattice)
    if tflag == "notransp":
        return np.concatenate([SCR1L_FFT(phat, I, lat, "notransp") for lat in lattices])
    elif tflag == "transp":
        out = 0
        start = 0
        for lat in lattices:
            out = out + SCR1L_FFT(phat[start : start + lat["M"]], I, lat, "transp")
            start += lat["M"]
        return out
    elif tflag == "notransp-transp":
        return sum(SCR1L_FFT(phat, I, lat, "notransp-transp") for lat in lattices)
    else:
        raise ValueError("unknown tflag")


def multi_SCR1L_nodes(multi_lattice, flag="cheby"):
    """
    Nodes of a union of shifted cosine rank-1 lattices, stacked in the order of the lattices.

    Parameters
    ----------
    multi_lattice : dict
        Dictionary with `zs`, `Ms` and `deltas`.
    flag : {'cheby', 'torus'}, optional
        Chebyshev nodes in [-1,1]^d (default) or lattice nodes in [0,1)^d, see `SCR1L_nodes`.

    Returns
    -------
    X : ndarray
        The sum(Ms) nodes, shape (sum(Ms), d).
    """
    return np.vstack(
        [
            SCR1L_nodes({"z": z, "M": M, "delta": delta}, flag)
            for z, M, delta in zip(multi_lattice["zs"], multi_lattice["Ms"], multi_lattice["deltas"])
        ]
    )


def multi_SCR1L_reconstruct(f, I, multi_lattice, tol=1e-10, wflag="unweighted", maxiter=None, return_info=False):
    """
    Least squares approximation from the samples on a union of shifted cosine rank-1 lattices.

    The samples of lattice r are transformed once by an FFT of length M_r, and LSQR is applied to the
    stacked problems sqrt(M_r) W_r (B_r(t_r) V y - ghat_r) on the residues l = 0, ..., floor(M_r/2), so
    that every iteration applies only the aliasing matrices, without FFT.

    Parameters
    ----------
    f : ndarray
        Sampling values at the nodes `multi_SCR1L_nodes(multi_lattice)`, shape (sum(Ms),).
    I : ndarray
        Index set of Chebyshev polynomials, shape (|I|, d).
    multi_lattice : dict
        Dictionary with `zs`, `Ms`, `deltas` and, optionally, `reco_infos`.
    tol : float, optional
        Tolerance `atol = btol` of LSQR. Default is 1e-10.
    wflag : {'unweighted', 'weighted'}, optional
        - `'unweighted'` (default): ordinary least squares on all samples in the orthonormal basis.
        - `'weighted'`: row weights H_r(l)^{-1/2} of every lattice and the column weights
          (sum_r M_r omega_{r,h})^{-1/2}, cf. Algorithm 6.1 of Potts and Ullrich.
    maxiter : int, optional
        Maximal number of LSQR iterations.
    return_info : bool, optional
        If True, also return a dict with the number of iterations.

    Returns
    -------
    f_hat : ndarray
        Coefficients in the orthonormal Chebyshev basis.
    info : dict, optional
        `iterations` (int).
    """
    lattices = _lattices(I, multi_lattice)
    f = np.asarray(f).reshape(-1)
    s = lattices[0]["reco_infos"]["nnz_s"]
    parts, ghats = [], []
    if wflag == "unweighted":
        v = 2.0 ** (s / 2)
    elif wflag == "weighted":
        omega = 0
    else:
        raise ValueError("unknown wflag")
    start = 0
    for lat in lattices:
        M = int(lat["M"])
        weights = SCR1L_weights(I, lat, wflag)
        if wflag == "weighted":
            omega = omega + M / weights["v"] ** 2
        parts.append((lat["reco_infos"], M, weights["w"], np.sqrt(M)))
        ghats.append(fft(f[start : start + M]) / M)
        start += M
    if wflag == "weighted":
        v = 1 / np.sqrt(omega)
    X = _FreqLS(parts, v)
    out = lsqr(X.linop(), X.rhs(ghats), atol=tol, btol=tol, iter_lim=maxiter)
    f_hat = v * out[0] / 2.0 ** (s / 2)
    if return_info:
        return f_hat, {"iterations": out[2]}
    return f_hat


def _lattices(I, multi_lattice):
    # per-lattice dictionaries; the aliasing structures are computed once and stored in `multi_lattice`
    if "reco_infos" not in multi_lattice:
        multi_lattice["reco_infos"] = [
            build_aliasing_structure(I, z, M, delta)
            for z, M, delta in zip(multi_lattice["zs"], multi_lattice["Ms"], multi_lattice["deltas"])
        ]
    return [
        {"z": z, "M": int(M), "delta": delta, "reco_infos": ri}
        for z, M, delta, ri in zip(
            multi_lattice["zs"], multi_lattice["Ms"], multi_lattice["deltas"], multi_lattice["reco_infos"]
        )
    ]
