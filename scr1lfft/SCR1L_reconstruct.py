import numpy as np
from scipy.fft import fft
from scipy.sparse.linalg import LinearOperator, lsqr
from scr1lfft.SCR1L_FFT import residues, residues_T, _get_reco_infos
from scr1lfft.SCR1L_weights import SCR1L_weights


def SCR1L_reconstruct(f, I, lattice, tol=1e-10, wflag="weighted", maxiter=None, return_info=False):
    """
    Least squares approximation from the samples on a shifted cosine rank-1 lattice (Algorithm 5.3).

    The samples are transformed once by an FFT of length M. The weighted least squares problem
        min_y || W (B(t) V y - ghat) ||_2,  ghat = fft(f) / M,
    is solved by LSQR on the residues l = 0, ..., floor(M/2) (factor sqrt(2) for 0 < l < M/2, real and
    imaginary parts stacked), so that every iteration applies only B(t) and B(t)^T, without FFT.

    Parameters
    ----------
    f : ndarray
        Sampling values at the nodes `SCR1L_nodes(lattice)`, shape (M,).
    I : ndarray
        Index set of Chebyshev polynomials, shape (|I|, d).
    lattice : dict
        Dictionary with `z`, `M`, `delta` and, optionally, `reco_infos`.
    tol : float, optional
        Tolerance `atol = btol` of LSQR. Default is 1e-10.
    wflag : {'weighted', 'unweighted'}, optional
        Weights of `SCR1L_weights`. Default is 'weighted'.
    maxiter : int, optional
        Maximal number of LSQR iterations.
    return_info : bool, optional
        If True, also return a dict with the number of iterations.

    Returns
    -------
    f_hat : ndarray
        Coefficients in the orthonormal Chebyshev basis psi_h = 2^{s(h)/2} phi_h.
    info : dict, optional
        `iterations` (int) and `weights` (dict of `SCR1L_weights`).
    """
    reco_infos = _get_reco_infos(I, lattice)
    M = int(lattice["M"])
    weights = SCR1L_weights(I, lattice, wflag)
    X = _FreqLS([(reco_infos, M, weights["w"], 1.0)], weights["v"])
    ghat = fft(np.asarray(f).reshape(-1)) / M
    out = lsqr(X.linop(), X.rhs([ghat]), atol=tol, btol=tol, iter_lim=maxiter)
    f_hat = weights["v"] * out[0] / 2.0 ** (reco_infos["nnz_s"] / 2)
    if return_info:
        return f_hat, {"iterations": out[2], "weights": weights}
    return f_hat


class _FreqLS:
    """Stacked real operator of the rows sqrt(M_r) s_r W_r B_r V of one or several lattices."""

    def __init__(self, parts, v):
        self.parts = []
        self.v = v
        rows = 0
        for reco_infos, M, w, s in parts:
            half = np.arange(M // 2 + 1)
            scale = np.full(len(half), np.sqrt(2.0))
            scale[0] = 1.0
            if M % 2 == 0:
                scale[-1] = 1.0
            sel = w[half] > 0
            keep = half[sel]
            wk = w[keep] * scale[sel] * s
            self.parts.append((reco_infos, M, keep, wk, rows))
            rows += 2 * len(keep)
        self.shape = (rows, len(v))

    def mat(self, Y):
        out = []
        C = self.v[:, None] * Y
        for reco_infos, M, keep, wk, _ in self.parts:
            R = residues(reco_infos, M, C)[keep] * wk[:, None]
            out += [R.real, R.imag]
        return np.vstack(out)

    def rmat(self, Z):
        acc = np.zeros((len(self.v), Z.shape[1]))
        for reco_infos, M, keep, wk, r0 in self.parts:
            k = len(keep)
            U = np.zeros((M, Z.shape[1]), dtype=np.complex128)
            U[keep] = wk[:, None] * (Z[r0 : r0 + k] - 1j * Z[r0 + k : r0 + 2 * k])
            acc += np.real(residues_T(reco_infos, U))
        return self.v[:, None] * acc

    def rhs(self, ghats):
        out = []
        for (reco_infos, M, keep, wk, _), ghat in zip(self.parts, ghats):
            r = ghat[keep] * wk
            out += [r.real, r.imag]
        return np.concatenate(out)

    def linop(self):
        return LinearOperator(
            self.shape,
            matvec=lambda y: self.mat(y.reshape(-1, 1))[:, 0],
            rmatvec=lambda b: self.rmat(b.reshape(-1, 1))[:, 0],
            dtype=float,
        )

    def gram(self):
        n = self.shape[1]
        f = lambda Y: self.rmat(self.mat(Y))
        return LinearOperator(
            (n, n), matvec=lambda y: f(y.reshape(n, -1)).reshape(y.shape), matmat=f, dtype=float
        )

    def dense(self):
        # only the rows of the residues hit by some mirror image are formed
        out = []
        for reco_infos, M, keep, wk, _ in self.parts:
            B = reco_infos["stages"][0]
            for St in reco_infos["stages"][1:]:
                B = St @ B
            R = B.tocsr()[np.searchsorted(reco_infos["rho"], keep)].toarray() * wk[:, None] * self.v[None, :]
            out += [R.real, R.imag]
        return np.vstack(out)
