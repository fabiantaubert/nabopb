import math
import warnings
import numpy as np
from scipy.sparse.linalg import lobpcg
from scr1lfft.build_aliasing_structure import build_aliasing_structure
from scr1lfft.divisor_chain import divisor_chain, grow_chain
from scr1lfft.SCR1L_weights import SCR1L_weights
from scr1lfft.SCR1L_reconstruct import _FreqLS


def SCR1L_search(index_set_I, c=1.25, K=6, m0=1, rng=None, dense_max=1500, extra=4, maxgrow=30):
    """
    Component-by-component construction of a shifted nested cosine rank-1 lattice with coset capacity
    (Algorithm 6.5 of D. Potts and T. Ullrich, Cosine approximation on shifted rank-1 lattices: sampling and
    arithmetic complexity without the factor 2^d).

    The divisor chain 1 = p_0 | p_1 | ... | p_d = M of `divisor_chain` defines the nested generating
    vector z_j = (M / p_j) w_j with units w_j in Z_{p_j}^*. In step j, random units w_j are admissible if
    no coset of a lattice subgroup of a stage m0 <= m < j is overloaded in the prefix problem with the
    index set P_{<=j} I. For up to K admissible units a shift Delta_j is drawn, and the pair with the
    smallest condition number of the weighted prefix problem (weights of Algorithm 6.1) is kept. If no
    admissible unit is found, the ratio p_j / p_{j-1} is enlarged.

    The construction is designed for lower sets I, for which the prefix problems are the problems of
    the projected index sets. For other index sets it is a heuristic; the condition number of the final
    problem is returned in `reco_infos['cond']`, and `multiSCR1L_search` gives a certified alternative.

    Parameters
    ----------
    index_set_I : ndarray
        Index set of shape (|I|, d) with nonnegative entries.
    c : float, optional
        Reserve factor of the divisor chain. Default is 1.25.
    K : int, optional
        Number of candidate units per component. Default is 6.
    m0 : int, optional
        First stage of the capacity test. Default is 1.
    rng : numpy.random.Generator, optional
        Random number generator.
    dense_max : int, optional
        Prefix problems with more indices are conditioned by LOBPCG instead of a dense matrix.
    extra : int, optional
        If none of the first K units is admissible, extra * K further units are screened.
    maxgrow : int, optional
        Maximal number of enlargements of one ratio of the chain.

    Returns
    -------
    z : ndarray
        Generating vector.
    M : int
        Lattice size.
    delta : ndarray
        Shift in [0,1)^d.
    reco_infos : dict
        Aliasing structure of `build_aliasing_structure` for `index_set_I`, together with
        `p` (the divisor chain) and `cond` (condition number of the final weighted problem).
    """
    I = np.asarray(index_set_I, dtype=np.int64)
    rng = np.random.default_rng() if rng is None else rng
    d = I.shape[1]
    p = divisor_chain(I, c)
    w = []
    delta = []
    for j in range(1, d + 1):
        J = np.unique(I[:, :j], axis=0)
        for _ in range(maxgrow + 1):
            pj = p[j - 1]
            best = (np.inf, None, None)
            pool = [int(u) for u in rng.integers(1, max(pj, 2), 8 * (extra + 1) * K) if math.gcd(int(u), pj) == 1]

            def lat(u):
                return np.array([(pj // p[i]) * (w + [u])[i] % pj for i in range(j)], dtype=np.int64)

            ok = [u for u in pool[:K] if _max_load_ratio(J, lat(u), pj, p, j, m0) <= 1.0]
            if not ok:
                ok = [u for u in pool[K : (extra + 1) * K] if _max_load_ratio(J, lat(u), pj, p, j, m0) <= 1.0][:K]
            for u in ok:
                dm = float(rng.random())
                lattice = {"z": lat(u), "M": pj, "delta": np.array(delta + [dm])}
                lattice["reco_infos"] = build_aliasing_structure(J, lattice["z"], pj, lattice["delta"])
                cd = _condition(J, lattice, dense_max)
                if cd < best[0]:
                    best = (cd, u, dm)
            if best[1] is not None:
                break
            p = grow_chain(p, j)
        if best[1] is None:
            raise RuntimeError("SCR1L_search: no admissible unit found")
        w.append(best[1])
        delta.append(best[2])

    M = int(p[-1])
    z = np.array([(M // p[i]) * w[i] % M for i in range(d)], dtype=np.int64)
    delta = np.array(delta)
    reco_infos = build_aliasing_structure(I, z, M, delta)
    reco_infos["p"] = p
    reco_infos["cond"] = best[0]
    return z, M, delta, reco_infos


def _max_load_ratio(J, zp, pj, p, j, m0):
    # largest load of a coset of (p_j / p_m) Z_{p_j}, m0 <= m < j, relative to its capacity p_m
    out = 0.0
    for m in range(m0, j):
        g = pj // p[m - 1]
        r = (J[:, m:] @ zp[m:]) % g
        out = max(out, np.bincount(r, minlength=g).max() / p[m - 1])
    return out


def _condition(J, lattice, dense_max):
    # condition number of the prefix problem with the weights of Algorithm 6.1
    weights = SCR1L_weights(J, lattice, "weighted")
    X = _FreqLS([(lattice["reco_infos"], lattice["M"], weights["w"], 1.0)], weights["v"])
    if X.shape[0] < X.shape[1]:
        return np.inf
    if J.shape[0] <= dense_max:
        D = X.dense()
        ev = np.linalg.eigvalsh(D.T @ D)
        return np.sqrt(ev[-1] / ev[0]) if ev[0] > ev[-1] * 1e-30 else np.inf
    return condition_estimate(X)


def condition_estimate(X, p=4, maxiter=80, tol=1e-5, rng=None):
    """Condition number of the operator X from the extreme eigenvalues of X^T X (LOBPCG)."""
    rng = np.random.default_rng(0) if rng is None else rng
    G = X.gram()
    n = X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lt, _ = lobpcg(G, rng.standard_normal((n, p)), largest=True, maxiter=maxiter, tol=tol)
        lb, _ = lobpcg(G, rng.standard_normal((n, p)), largest=False, maxiter=maxiter, tol=tol)
    lmin = np.min(lb)
    return np.sqrt(np.max(lt) / lmin) if lmin > 0 else np.inf
