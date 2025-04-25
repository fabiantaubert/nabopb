import numpy as np
import sympy as sy

def heuristic_lattice_search(I):
    """
    Compute a generating lattice for a given frequency index set.

    Parameters
    ----------
    I : array_like
        Frequency index set, where `I` is a 2D array with shape `(n, d)`.

    Returns
    -------
    z : ndarray
        The generating vector `z`, an array of integers.
    M : int
        The lattice size.
    """
    
    Mtmp = sy.nextprime(max(I.shape[0]**2, np.max(I) - np.min(I) + 1)-1) # -1 to check if val is already prime
    T = 100
    numtests = 0
    M = -1
    z = np.zeros(I.shape[1])

    I = I[np.lexsort(I.T[::-1])]

    while numtests < 5:
        ztmp, itno = search_lattice_onlyT(I, Mtmp, T)
        if np.min(itno) > -1:
            M = Mtmp
            z = np.mod(ztmp, Mtmp)
            if Mtmp == 2:
                break
            Mtmp = sy.nextprime(Mtmp/2)
            numtests = -1
        numtests += 1

    return z, M

def search_lattice_onlyT(I, M, T):
    maxd = I.shape[1]
    z = np.zeros(maxd)
    z[0] = 1
    tmp1 = z[0] * I[:, 0]
    search_itno = np.ones([maxd,1]) * -1
    search_itno[0] = 0

    for d in range(1, maxd):
        if search_itno[d-1] == -1:
            break
        # Recognize duplicates
        I2tmp = np.abs(tmp1[1:] - tmp1[:-1]) + np.abs(I[1:, d] - I[:-1, d])
        ind = np.where(I2tmp != 0)[0]
        ind = np.append(ind, len(tmp1)-1)
        inds = np.zeros(len(ind), dtype=int)
        zaehler1 = 0
        zaehler2 = len(ind)
        for j in range(len(ind)):
            if I[ind[j], d] == 0:
                inds[zaehler2-1] = ind[j]
                zaehler2 -= 1
            else:
                inds[zaehler1] = ind[j]
                zaehler1 += 1

        I21a = tmp1[inds[:zaehler2]]
        I22a = I[inds[:zaehler2], d]
        checkvar = np.zeros(len(ind))
        checkvar[zaehler2:] = tmp1[inds[zaehler2:]]

        if T < M:
            zcomp_cands = np.random.choice(M, size=T, replace=False) + 1
        else:
            zcomp_cands = np.random.permutation(np.arange(1, M+1))
            T = M

        for r in range(T):
            checkvar[:zaehler2] = np.mod(I21a + I22a * zcomp_cands[r], M)
            if len(np.unique(checkvar)) == len(checkvar):
                z[d] = zcomp_cands[r]
                tmp1 = np.mod(tmp1 + I[:, d] * z[d], M)
                search_itno[d] = r
                break

    return z, search_itno