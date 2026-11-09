import numpy as np
from scipy.sparse import csr_matrix


def build_aliasing_structure(I, z, M, delta=None, flag="auto"):
    """
    Precomputes the aliasing matrix B(t) of a shifted rank-1 lattice for the cosine basis.

    For the nodes x_k = (k z / M + delta) mod 1, k = 0, ..., M-1, the sampling matrix of the cosine
    functions phi_h(x) = prod_j cos(2 pi h_j x_j), h in I, factorizes as A = F_M B(t), where
    F_M = (exp(2 pi i k l / M))_{k,l} and

        B_{l,h}(t) = 2^{-s(h)} sum_{eps in E(h), (eps o h) . z = l mod M} t^{eps o h},
        t = (exp(2 pi i delta_j))_j,

    with the number s(h) of nonzero entries of h and the sign vectors E(h) on the support of h.
    B is stored as a product B = P S_d ... S_1 of sparse stages, where P places the rows of the
    last stage at the residues `rho`.

    Parameters
    ----------
    I : ndarray
        Index set of shape (|I|, d) with nonnegative entries.
    z : ndarray
        Generating vector of length d.
    M : int
        Lattice size.
    delta : ndarray, optional
        Shift in [0,1)^d. Default is no shift.
    flag : {'auto', 'staged', 'direct'}, optional
        - `'staged'`: d stages that merge the pairs (suffix of h, partial residue), cf. the staged
          application for nested lattices. Efficient for lower sets such as hyperbolic crosses.
        - `'direct'`: one stage with one entry per mirror image eps o h, i.e., sum_h 2^{s(h)}
          nonzeros. Efficient for sparse index sets.
        - `'auto'` (default): the variant with fewer nonzeros.

    Returns
    -------
    reco_infos : dict
        - `stages` (list of csr_matrix): the sparse stages S_1, ..., S_d (only S_1 for 'direct').
        - `patterns` (list of csr_matrix): the same stages with all phases replaced by 1.
        - `rho` (ndarray): residues of the rows of the last stage.
        - `ops` (int): total number of nonzeros of the stages.
        - `nnz_s` (ndarray): number s(h) of nonzero entries of every h in I.
    """
    I = np.asarray(I, dtype=np.int64)
    z = np.asarray(z, dtype=np.int64).reshape(-1)
    M = int(M)
    if np.min(I) < 0:
        raise ValueError("input index set must not contain frequencies less than zero")
    if flag not in ["auto", "staged", "direct"]:
        raise ValueError("unknown flag")

    mu = int(np.sum(2 ** np.sum(I != 0, axis=1)))
    if flag == "direct":
        reco_infos = _direct_stage(I, z, M, delta)
    else:
        reco_infos = _staged(I, z, M, delta)
        if flag == "auto" and reco_infos["ops"] > mu:
            reco_infos = _direct_stage(I, z, M, delta)
    reco_infos["nnz_s"] = np.sum(I != 0, axis=1)
    return reco_infos


def _suffix_structure(I):
    # unique suffixes I[:, m:] for m = 0, ..., d and the map of every suffix to its tail
    d = I.shape[1]
    S = []
    inv = []
    for m in range(d + 1):
        if m < d:
            u, iv = np.unique(I[:, m:], axis=0, return_inverse=True)
        else:
            u, iv = np.zeros((1, 0), dtype=np.int64), np.zeros(I.shape[0], dtype=np.int64)
        S.append(u)
        inv.append(iv.reshape(-1))
    par = [None]
    for m in range(1, d + 1):
        p = np.zeros(len(S[m - 1]), dtype=np.int64)
        p[inv[m - 1]] = inv[m]
        par.append(p)
    return S, inv, par


def _staged(I, z, M, delta):
    S, inv, par = _suffix_structure(I)
    d = I.shape[1]
    sid = np.arange(len(S[0]))
    rho = np.zeros(len(sid), dtype=np.int64)
    stages = []
    patterns = []
    for m in range(1, d + 1):
        hm = S[m - 1][sid, 0]
        p = par[m][sid]
        a = hm * z[m - 1] % M
        nz = hm != 0
        if delta is None:
            ph = np.ones(len(sid))
        else:
            ph = np.exp(2j * np.pi * hm * delta[m - 1])
        src = np.concatenate([np.arange(len(sid)), np.nonzero(nz)[0]])
        new_suffix = np.concatenate([p, p[nz]])
        new_res = np.concatenate([(rho + a) % M, (rho - a)[nz] % M])
        w = np.concatenate([np.where(nz, 0.5 * ph, 1.0), 0.5 * np.conj(ph[nz])])
        key, tgt = np.unique(new_suffix * M + new_res, return_inverse=True)
        tgt = tgt.reshape(-1)
        shape = (len(key), len(sid))
        stages.append(csr_matrix((w, (tgt, src)), shape=shape))
        patterns.append(csr_matrix((np.ones(len(w)), (tgt, src)), shape=shape))
        sid = key // M
        rho = key % M
    # the columns of the first stage are the sorted unique rows S[0]; map them to the rows of I
    stages[0] = stages[0][:, inv[0]]
    patterns[0] = patterns[0][:, inv[0]]
    return {
        "stages": stages,
        "patterns": patterns,
        "rho": rho,
        "ops": int(sum(St.nnz for St in stages)),
    }


def _direct_stage(I, z, M, delta):
    n, d = I.shape
    rows = [np.zeros(n, dtype=np.int64)]  # partial residues of the mirror images
    phase = [np.zeros(n)]  # partial values of (eps o h) . delta
    cols = [np.arange(n)]
    for j in range(d):
        new_rows, new_phase, new_cols = [], [], []
        for r, f, c in zip(rows, phase, cols):
            h = I[c, j]
            dj = 0.0 if delta is None else delta[j]
            new_rows.append((r + h * z[j]) % M)
            new_phase.append(f + h * dj)
            new_cols.append(c)
            nz = h != 0
            new_rows.append((r[nz] - h[nz] * z[j]) % M)
            new_phase.append(f[nz] - h[nz] * dj)
            new_cols.append(c[nz])
        rows = [np.concatenate(new_rows)]
        phase = [np.concatenate(new_phase)]
        cols = [np.concatenate(new_cols)]
    rows, phase, cols = rows[0], phase[0], cols[0]
    s = np.sum(I != 0, axis=1)
    rho, tgt = np.unique(rows, return_inverse=True)
    tgt = tgt.reshape(-1)
    if delta is None:
        w = 2.0 ** (-s[cols])
    else:
        w = 2.0 ** (-s[cols]) * np.exp(2j * np.pi * phase)
    shape = (len(rho), n)
    stage = csr_matrix((w, (tgt, cols)), shape=shape)
    pattern = csr_matrix((np.ones(len(w)), (tgt, cols)), shape=shape)
    return {"stages": [stage], "patterns": [pattern], "rho": rho, "ops": int(stage.nnz)}
