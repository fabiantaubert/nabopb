import numpy as np
from scipy.fft import rfft, irfft
from scr1lfft.build_aliasing_structure import build_aliasing_structure


def SCR1L_FFT(phat, I, lattice, tflag="notransp"):
    """
    Fast evaluation of Chebyshev polynomials on a shifted cosine rank-1 lattice and its transpose.

    The nodes are x_k = (k z / M + delta) mod 1, k = 0, ..., M-1, in [0,1)^d, i.e., the Chebyshev nodes
    cos(2 pi x_k) in [-1,1]^d, see `SCR1L_nodes`. The polynomials are given in the orthonormal Chebyshev
    basis psi_h(x) = 2^{s(h)/2} prod_j cos(2 pi h_j x_j), as in `cmr1lfft`. One application costs the
    nonzeros of the aliasing stages (`reco_infos['ops']`) and one FFT of length M for real data: B(t) c is
    conjugate symmetric for real c, so A c = F_M B(t) c is the inverse FFT with real output (`irfft`) of the
    residues l = 0, ..., floor(M/2), and A^T y = B(t)^T F_M^T y uses the FFT for real input (`rfft`).

    Parameters
    ----------
    phat : ndarray
        The input coefficients or sampling values depending on the value of `tflag`:
        - For `tflag='notransp'` and `tflag='notransp-transp'`, `phat` contains the Chebyshev
          coefficients, shape (|I|,) or (|I|, r).
        - For `tflag='transp'`, `phat` contains the sampling values, shape (M,) or (M, r).
    I : ndarray
        Index set of Chebyshev polynomials, a matrix of shape (|I|, d), where |I| is the number of indices
        and `d` is the spatial dimension. Each row corresponds to a multi-index.
    lattice : dict
        Dictionary containing the shifted lattice:
        - `z` (ndarray): Generating vector.
        - `M` (int): Lattice size.
        - `delta` (ndarray): Shift in [0,1)^d.
        - `reco_infos` (dict, optional): Aliasing structure of `build_aliasing_structure`. It is
          computed and stored in `lattice` if not provided.
    tflag : {'notransp', 'transp', 'notransp-transp'}, optional
        Indicates the type of operation to perform (default: 'notransp'):
        - `'notransp'`: Compute the sampling values A phat.
        - `'transp'`: Compute the transpose A^T phat.
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
    reco_infos = _get_reco_infos(I, lattice)
    M = int(lattice["M"])
    scal = 2.0 ** (reco_infos["nnz_s"] / 2)
    vec = np.ndim(phat) == 1
    # A is real; only the real part of complex input is used
    X = np.real(np.asarray(phat)).reshape(len(phat), -1)

    if tflag == "notransp":
        # B(t) X is conjugate symmetric, irfft uses its rows l = 0, ..., floor(M/2)
        G = residues(reco_infos, M, scal[:, None] * X)
        p = M * irfft(G[: M // 2 + 1], n=M, axis=0)
    elif tflag == "transp":
        # F_M^T X = conj(fft(X)) has the rows conj(R_l) for l <= M/2 and R_{M-l} for l > M/2, R = rfft(X)
        R = rfft(X, axis=0)
        h = M // 2 + 1
        U = np.empty((M, X.shape[1]), dtype=R.dtype)
        np.conjugate(R, out=U[:h])
        U[h:] = R[M - h : 0 : -1]
        p = scal[:, None] * np.real(residues_T(reco_infos, U))
    elif tflag == "notransp-transp":
        G = residues(reco_infos, M, scal[:, None] * X)
        p = M * scal[:, None] * np.real(residues_T(reco_infos, np.conj(G)))
    else:
        raise ValueError("unknown tflag")

    return p[:, 0] if vec else p


def SCR1L_nodes(lattice, flag="cheby"):
    """
    Nodes of a shifted cosine rank-1 lattice.

    Parameters
    ----------
    lattice : dict
        Dictionary with the generating vector `z`, the lattice size `M` and the shift `delta`.
    flag : {'cheby', 'torus'}, optional
        - `'cheby'` (default): the Chebyshev nodes cos(2 pi x_k) in [-1,1]^d.
        - `'torus'`: the lattice nodes x_k = (k z / M + delta) mod 1 in [0,1)^d.

    Returns
    -------
    X : ndarray
        The M nodes, shape (M, d), in the order k = 0, ..., M-1 used by `SCR1L_FFT`.
    """
    z = np.asarray(lattice["z"], dtype=np.int64).reshape(-1)
    M = int(lattice["M"])
    delta = np.zeros(len(z)) if lattice.get("delta") is None else np.asarray(lattice["delta"])
    X = np.mod(np.mod(np.arange(M).reshape(-1, 1) * z, M) / M + delta, 1.0)
    if flag == "cheby":
        return np.cos(2 * np.pi * X)
    elif flag == "torus":
        return X
    else:
        raise ValueError("unknown flag")


def residues(reco_infos, M, C):
    """B(t) C for the coefficients C in the cosine basis phi_h, a matrix of shape (M, r)."""
    V = C
    for St in reco_infos["stages"]:
        V = St @ V
    G = np.zeros((M, C.shape[1]), dtype=np.complex128)
    G[reco_infos["rho"]] = V
    return G


def residues_T(reco_infos, U):
    """B(t)^T U (transpose without conjugation) for U of shape (M, r)."""
    V = U[reco_infos["rho"]]
    for St in reversed(reco_infos["stages"]):
        V = St.T @ V
    return V


def hit_numbers(reco_infos, M, n):
    """Hit numbers H(l) = #{(h, eps): (eps o h) . z = l mod M}, l = 0, ..., M-1."""
    v = np.ones(n)
    for P in reco_infos["patterns"]:
        v = P @ v
    H = np.zeros(M)
    H[reco_infos["rho"]] = v
    return H


def orbit_sum(reco_infos, f):
    """sum_{eps in E(h)} f((eps o h) . z mod M) for every h, for a vector f on the residues."""
    v = f[reco_infos["rho"]]
    for P in reversed(reco_infos["patterns"]):
        v = P.T @ v
    return v


def _get_reco_infos(I, lattice):
    if "reco_infos" not in lattice:
        lattice["reco_infos"] = build_aliasing_structure(
            I, lattice["z"], lattice["M"], lattice.get("delta")
        )
    return lattice["reco_infos"]
