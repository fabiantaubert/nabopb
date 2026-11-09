import math
import numpy as np
from sympy import nextprime


def divisor_chain(I, c=1.25, p1min=0, maxit=500):
    """
    Divisor chain with live capacity (Algorithm 5.1).

    Computes 1 = p_0 | p_1 | ... | p_d = M with prime ratios q_m = p_m / p_{m-1} such that
    p_m >= c n_m with n_m = |P_{<=m} I| (the fine chain), and enlarges the chain until the necessary
    condition p_m >= c L_m holds for the numbers L_m of indices whose mirror images all lie in the
    subgroup (M / p_m) Z_M.

    Parameters
    ----------
    I : ndarray
        Lower set of shape (|I|, d) with nonnegative entries.
    c : float, optional
        Reserve factor c >= 1. Default is 1.25.
    p1min : int, optional
        Lower bound for p_1. With shift p_1 >= N_1 + 1 suffices, where N_1 is the largest first entry.
    maxit : int, optional
        Maximal number of enlargements.

    Returns
    -------
    p : list of int
        The divisor chain p_1, ..., p_d with M = p_d.
    """
    n = prefix_sizes(I)
    d = len(n)
    p = [int(nextprime(max(math.ceil(c * n[0]), p1min) - 1))]
    for m in range(1, d):
        if p[-1] >= c * n[m]:
            p.append(p[-1])
        else:
            p.append(p[-1] * int(nextprime(math.ceil(c * n[m] / p[-1]) - 1)))
    for _ in range(maxit):
        L = live_counts(I, p)
        bad = [m for m in range(d) if p[m] < c * L[m]]
        if not bad:
            return [int(x) for x in p]
        p = grow_chain(p, bad[0] + 1)
    raise RuntimeError("divisor_chain: live capacity not reached")


def prefix_sizes(I):
    """n_m = |P_{<=m} I|, m = 1, ..., d."""
    return [len(np.unique(I[:, :m], axis=0)) for m in range(1, I.shape[1] + 1)]


def live_counts(I, p):
    """
    L_m = #{h in I: all mirror images of h lie in (M / p_m) Z_M}, m = 1, ..., d, independent of the
    units: (p_j / p_m) | 2 h_j for all j > m, and the number of j > m with (p_j / p_m) not dividing h_j
    is even.
    """
    d = I.shape[1]
    out = []
    for m in range(1, d + 1):
        ok = np.ones(I.shape[0], dtype=bool)
        odd = np.zeros(I.shape[0], dtype=np.int64)
        for j in range(m, d):
            r = p[j] // p[m - 1]
            ok &= (2 * I[:, j]) % r == 0
            odd += I[:, j] % r != 0
        out.append(int((ok & (odd % 2 == 0)).sum()))
    return out


def grow_chain(p, j):
    """Replace the ratio q_j by the smallest prime >= 3 larger than q_j; all other ratios are kept."""
    q = [p[0]] + [p[m] // p[m - 1] for m in range(1, len(p))]
    q[j - 1] = int(nextprime(max(q[j - 1], 2)))
    return [int(x) for x in np.cumprod(q, dtype=np.int64)]
