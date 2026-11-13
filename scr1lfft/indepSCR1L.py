import numpy as np
from sympy import nextprime
from scipy.sparse.linalg import LinearOperator, lsqr
from scr1lfft.SCR1L_FFT import SCR1L_FFT


def indepSCR1L_search(I, L=8, rng=None):
    """
    Independent random shifted cosine rank-1 lattices and the Christoffel function on their nodes.

    The L lattices have the smallest prime size M >= max(mu, 2 N_inf + 1), where mu = sum_h 2^{s(h)} is the size
    of the mirrored set and N_inf the largest entry of I, and generating vectors and shifts uniformly distributed
    in {0, ..., M-1}^d and [0,1)^d. For every finite index set, L >= 13.1 ln(|I|/eta) such lattices with the row
    weights H_r(l)^{-1/2} have a condition number smaller than 2 with probability at least 1 - eta, and draws
    proportional to the Christoffel function K = sum_h psi_h^2 from their nodes reduce the number of samples to
    O(H |I| log |I|) (Theorem 4.14 and Corollary 4.15 of D. Potts, Cosine approximation on shifted nested rank-1
    lattices without the factor 2^d). K is evaluated on every lattice by one application of `SCR1L_FFT` to the
    expansion sum_h psi_h^2 = sum_g a_g psi_g with g = 2 h_T, T a subset of the support of h.

    Parameters
    ----------
    I : ndarray
        Index set of Chebyshev polynomials, shape (|I|, d), nonnegative entries.
    L : int, optional
        Number of lattices. Default is 8.
    rng : numpy.random.Generator, optional
        Random number generator. Default is a generator seeded from the global state of `np.random`.

    Returns
    -------
    pool : dict
        - `zs` (ndarray): Generating vectors, shape (L, d).
        - `Ms` (ndarray): Lattice sizes (all equal to M).
        - `deltas` (ndarray): Shifts, shape (L, d).
        - `christoffel` (ndarray): K at the nodes of all lattices, shape (L, M), nodes as in `SCR1L_nodes`.
    """
    I = np.asarray(I, dtype=np.int64)
    rng = np.random.default_rng(np.random.randint(2**31)) if rng is None else rng
    n, d = I.shape
    mu = int(np.sum(2 ** np.count_nonzero(I, axis=1)))
    M = int(nextprime(max(mu, 2 * int(I.max()) + 1) - 1))
    G, a = _squares(I)
    zs = rng.integers(0, M, (L, d))
    deltas = rng.random((L, d))
    K = np.array([SCR1L_FFT(a, G, {"z": z, "M": M, "delta": delta}, "notransp") for z, delta in zip(zs, deltas)])
    return {"zs": zs, "Ms": np.full(L, M), "deltas": deltas, "christoffel": K}


def indep_SCR1L_subsample(pool, m, rng=None, accept=None):
    """
    Draws from the nodes of independent lattices with probabilities proportional to the Christoffel function.

    Parameters
    ----------
    pool : dict
        Output of `indepSCR1L_search`.
    m : int
        Number of draws (with replacement).
    rng : numpy.random.Generator, optional
        Random number generator. Default is a generator seeded from the global state of `np.random`.
    accept : callable, optional
        Function of the Chebyshev nodes (shape (k, d)) with values in [0, 1]. If given, the draws are proportional
        to accept(x) K(x) (rejection sampling), e.g. the Christoffel function of a subspace divided by K.

    Returns
    -------
    sub : dict
        - `lat` (ndarray): Lattice of every distinct drawn node.
        - `idx` (ndarray): Index k of the node x_k = (k z / M + delta) mod 1 in its lattice.
        - `w` (ndarray): Row weights (count / (m p))^{1/2} up to a common factor, p the probability of the node.
    """
    rng = np.random.default_rng(np.random.randint(2**31)) if rng is None else rng
    K = pool["christoffel"]
    L, M = K.shape
    p = K.ravel() / K.sum()
    if accept is None:
        sel = rng.choice(L * M, m, p=p)
    else:
        sel = np.zeros(0, dtype=np.int64)
        while len(sel) < m:
            prop = rng.choice(L * M, 2 * (m - len(sel)) + 16, p=p)
            sel = np.concatenate([sel, prop[rng.random(len(prop)) < accept(_nodes(pool, prop))]])
        sel = sel[:m]
    u, cnt = np.unique(sel, return_counts=True)
    q = p[u] if accept is None else p[u] * accept(_nodes(pool, u))
    return {"lat": u // M, "idx": u % M, "w": np.sqrt(cnt / (m * q))}


def indep_SCR1L_nodes(pool, sub, flag="cheby"):
    """
    The drawn nodes, the Chebyshev nodes cos(2 pi x) in [-1,1]^d (flag='cheby', default) or x in [0,1)^d ('torus').
    """
    X = _nodes(pool, sub["lat"] * pool["christoffel"].shape[1] + sub["idx"], "torus")
    if flag == "cheby":
        return np.cos(2 * np.pi * X)
    elif flag == "torus":
        return X
    else:
        raise ValueError("unknown flag")


def indep_SCR1L_reconstruct(f, I, pool, sub, affine=None, tol=1e-10, maxiter=None, return_info=False):
    """
    Weighted least squares approximation from the samples at the drawn nodes.

    Every LSQR iteration applies `SCR1L_FFT` on all lattices, i.e., the aliasing matrices and one FFT per lattice.

    Parameters
    ----------
    f : ndarray
        Sampling values at the nodes `indep_SCR1L_nodes(pool, sub)`.
    I : ndarray
        Index set of Chebyshev polynomials, shape (|I|, d).
    pool, sub : dict
        Outputs of `indepSCR1L_search` and `indep_SCR1L_subsample`.
    affine : tuple, optional
        (u0, N) with the coefficients restricted to u0 + range(N), N of shape (|I|, n') with orthonormal columns,
        e.g. from known linear constraints. Then only the n' coordinates of z in u = u0 + N z are fitted.
    tol : float, optional
        Tolerance `atol = btol` of LSQR. Default is 1e-10.
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
    I = np.asarray(I, dtype=np.int64)
    f = np.asarray(f).reshape(-1)
    w = sub["w"]
    M = pool["christoffel"].shape[1]
    lattices = pool.setdefault("lattices", [{"z": z, "M": M, "delta": delta} for z, delta in zip(pool["zs"], pool["deltas"])])
    parts = [np.flatnonzero(sub["lat"] == r) for r in range(len(lattices))]

    def A(c):
        y = np.empty(len(w))
        for lat, part in zip(lattices, parts):
            if len(part):
                y[part] = SCR1L_FFT(c, I, lat, "notransp")[sub["idx"][part]]
        return w * y

    def At(y):
        out = np.zeros(len(I))
        y = w * np.asarray(y).reshape(-1)
        for lat, part in zip(lattices, parts):
            if len(part):
                v = np.zeros(M)
                v[sub["idx"][part]] = y[part]
                out += SCR1L_FFT(v, I, lat, "transp")
        return out

    u0, N = (np.zeros(len(I)), None) if affine is None else affine
    rhs = w * f - (A(u0) if affine is not None else 0)
    if N is None:
        op = LinearOperator((len(w), len(I)), matvec=A, rmatvec=At, dtype=float)
    elif N.shape[1] == 0:
        return (u0, {"iterations": 0}) if return_info else u0
    else:
        op = LinearOperator((len(w), N.shape[1]), matvec=lambda z: A(N @ z), rmatvec=lambda y: N.T @ At(y), dtype=float)
    out = lsqr(op, rhs, atol=tol, btol=tol, iter_lim=maxiter)
    f_hat = out[0] if N is None else u0 + N @ out[0]
    if return_info:
        return f_hat, {"iterations": out[2]}
    return f_hat


def _squares(I):
    # index set G = {2 h_T} and coefficients a with sum_h psi_h^2 = sum_g a_g psi_g, since
    # psi_h^2 = prod_{j in supp h} (1 + cos(4 pi h_j x_j)) = sum_T 2^{-|T|/2} psi_{2 h_T}
    s = np.count_nonzero(I, axis=1)
    G, a = [], []
    for k in np.unique(s):
        rows = I[s == k]
        sup = np.argsort(rows == 0, axis=1, kind="stable")[:, :k]
        for mask in range(2**k):
            bits = np.array([(mask >> b) & 1 for b in range(k)], dtype=bool)
            g = np.zeros_like(rows)
            if bits.any():
                cols = sup[:, bits]
                np.put_along_axis(g, cols, 2 * np.take_along_axis(rows, cols, axis=1), axis=1)
            G.append(g)
            a.append(np.full(len(rows), 2.0 ** (-bits.sum() / 2)))
    G, inv = np.unique(np.vstack(G), axis=0, return_inverse=True)
    return G, np.bincount(inv.reshape(-1), weights=np.concatenate(a))


def _nodes(pool, flat, flag="cheby"):
    # nodes of the flat indices r M + k of all lattices
    M = pool["christoffel"].shape[1]
    r, k = np.divmod(np.asarray(flat), M)
    X = np.mod(np.mod(k[:, None] * pool["zs"][r], M) / M + pool["deltas"][r], 1.0)
    return np.cos(2 * np.pi * X) if flag == "cheby" else X
