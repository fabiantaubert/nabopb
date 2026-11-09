import math
import numpy as np
from sympy import nextprime
from scr1lfft.build_aliasing_structure import build_aliasing_structure


def multiSCR1L_search(index_set_I, c=1.0, eta=None, rng=None, pmin=11, max_lattices=200):
    """
    Union of shifted nested cosine rank-1 lattices with peeling (Algorithm 5.4).

    An index h is resolved by lattice r if its residue (eta o h) . z_r mod M_r contains no other
    unresolved index of I; resolved indices are removed from all lattices, and this is repeated as long as
    new indices are resolved (peeling across the lattices). Lattice r is a nested lattice for the
    unresolved set U with the fine chain for the targets max{c |P_{<=m} U|, max_U h_m - min_U h_m + 1}
    (the first target at least `pmin`). If the last three lattices resolved no index, the next lattice
    is a general rank-1 lattice with prime size M >= max{4 |U|, N_inf + 1} and uniform z. The lattices
    peel I, so the sampling matrix of the union has full rank for almost all shifts (Theorem 3.18); the
    shifts are drawn independently and uniformly. The construction works for every finite index set.

    Parameters
    ----------
    index_set_I : ndarray
        Index set of shape (|I|, d) with nonnegative entries.
    c : float, optional
        Reserve factor. Default is 1.0.
    eta : ndarray, optional
        Sign vector in {-1, 1}^d of the mirrored copy. Default is (1, ..., 1).
    rng : numpy.random.Generator, optional
        Random number generator.
    pmin : int, optional
        Lower bound for the first prime of the chain. Default is 11.
    max_lattices : int, optional
        Maximal number of lattices.

    Returns
    -------
    zs : ndarray
        Generating vectors, shape (R, d).
    Ms : ndarray
        Lattice sizes, shape (R,).
    deltas : ndarray
        Shifts in [0,1)^d, shape (R, d).
    reco_infos : list of dict
        Aliasing structures of `build_aliasing_structure` for `index_set_I`, one per lattice.
    """
    I = np.asarray(index_set_I, dtype=np.int64)
    rng = np.random.default_rng() if rng is None else rng
    n, d = I.shape
    eta = np.ones(d, dtype=np.int64) if eta is None else np.asarray(eta, dtype=np.int64)
    N_inf = int(np.max(I))
    resolved = np.zeros(n, dtype=bool)
    residue_lists = []
    zs, Ms, gains = [], [], []
    while not resolved.all():
        if len(zs) >= max_lattices:
            raise RuntimeError("multiSCR1L_search: maximal number of lattices reached")
        U = I[~resolved]
        if len(gains) >= 3 and sum(gains[-3:]) == 0:
            M = int(nextprime(max(4 * len(U), N_inf + 1) - 1))
            z = rng.integers(0, M, d).astype(np.int64)
        else:
            z, M = _nested_lattice(_fine_chain(U, c, pmin), rng)
        before = resolved.sum()
        residue_lists.append(((I * eta) @ z) % M)
        _peel(residue_lists, resolved)
        gains.append(resolved.sum() - before)
        zs.append(z)
        Ms.append(M)

    zs = np.array(zs, dtype=np.int64)
    Ms = np.array(Ms, dtype=np.int64)
    deltas = rng.random((len(Ms), d))
    reco_infos = [build_aliasing_structure(I, zs[r], Ms[r], deltas[r]) for r in range(len(Ms))]
    return zs, Ms, deltas, reco_infos


def _fine_chain(U, c, pmin):
    # p_m >= max{c |P_{<=m} U|, max_U h_m - min_U h_m + 1} with prime ratios, p_1 >= pmin
    d = U.shape[1]
    p = []
    prev = 1
    for m in range(1, d + 1):
        target = max(c * len(np.unique(U[:, :m], axis=0)), np.ptp(U[:, m - 1]) + 1, pmin if m == 1 else 0)
        if prev >= target:
            p.append(prev)
            continue
        q = int(nextprime(math.ceil(target / prev) - 1))
        prev *= q
        p.append(prev)
    return p


def _nested_lattice(p, rng):
    # z_j = (M / p_j) w_j with random units w_j in Z_{p_j}^*
    M = p[-1]
    z = []
    for pj in p:
        while True:
            w = int(rng.integers(1, pj)) if pj > 1 else 0
            if pj == 1 or math.gcd(w, pj) == 1:
                break
        z.append((M // pj) * w % M)
    return np.array(z, dtype=np.int64), int(M)


def _peel(residue_lists, resolved):
    # resolve every unresolved index that is alone among the unresolved ones in some residue class
    while True:
        new = np.zeros(len(resolved), dtype=bool)
        for r in residue_lists:
            cnt = np.bincount(r[~resolved], minlength=int(r.max()) + 1)
            new |= (~resolved) & (cnt[r] == 1)
        if not new.any():
            return resolved
        resolved |= new
