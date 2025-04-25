import numpy as np
from cmr1lfft import *
from sympy import nextprime, primerange

def multiCR1L_search(index_set_I, algno, MI=None, MI_index_I=None):
    """
    Executes the multiCR1L search process using a specified algorithm.

    Parameters
    ----------
    index_set_I : ndarray
        A 2D array representing the index set used for the search. Each row corresponds to an index vector.
    algno : int
        An integer specifying the algorithm to use:
        - 1: Simple random draw (`multiCR1L_simple_random_draw`)
        - 2: Simple greedy (`multiCR1L_simple_greedy`)
        - 3: Simple iterative decoding (`multiCR1L_simple_Idec`)
        - 4: Improved iterative decoding (`multiCR1L_improved_Idec`)
        - 21: Simple greedy with `waawM` modification (`multiCR1L_simple_greedy_waawM`)
        - 31: Simple iterative decoding with `waawM` modification (`multiCR1L_simple_Idec_waawM`)
        - 41: Improved iterative decoding with `waawM` modification (`multiCR1L_improved_Idec_waawM`)
    MI : ndarray, optional
        A precomputed mirrored index set. If not provided, it will be computed internally.
    MI_index_I : ndarray, optional
        Index mapping corresponding to the mirrored index set. If not provided, it will be computed internally.

    Returns
    -------
    zs : ndarray
        An array of generating vectors.
    Ms : ndarray
        A vector of lattice sizes.
    reco_infos : dict
        A dictionary containing additional reconstruction information:
        - 'M_I_index_I': The mirrored index mapping.
        - 'kz_mod_M': The modulo results for each generating vector and lattice size.

    Raises
    ------
    ValueError
        If an unsupported algorithm number is provided.
    """

    if MI is None:
        MI = []
        MI_index_I = []

    if algno == 1:
        # constructs multiCR1L according to theory with c=2 and probability delta=|I|^{-1} (simple random draw)
        zs, Ms = multiCR1L_simple_random_draw(index_set_I)
    elif algno == 2:
        # constructs multiCR1L according to theory with c=2, delta=|I|^{-1} and aliasing wrt. all h\in M(I);
        zs, Ms, reco_infos = multiCR1L_simple_greedy(index_set_I, MI, MI_index_I)
        # result should be better than in case 1 due to lower number of rank-1 lattices
    elif algno == 3:
        # constructs multiCR1L with aliasing wrt. all h\in M(I) and decreasing lattice sizes
        zs, Ms = multiCR1L_simple_Idec(index_set_I, None, MI, MI_index_I)
        # result should be better than in case 2 due to decreasing rank-1 lattices sizes
    elif algno == 4:
        # constructs multiCR1L with aliasing wrt. all h\in M(I) and decreasing lattice sizes
        zs, Ms, _ = multiCR1L_improved_Idec(index_set_I, None, MI, MI_index_I)
        # result should be better than in case 3 due to decreasing and 'optimized' rank-1 lattices sizes
    elif algno == 21:
        # constructs multiCR1L according to theory with c=2, delta=|I|^{-1} and aliasing wrt. all h\in M(I)
        zs, Ms, reco_infos = multiCR1L_simple_greedy_waawM(index_set_I)
        # should be not worse than case 2 in average due to waawM (with aliasing allowed within M(k))
    elif algno == 31:
        # constructs multiCR1L with aliasing wrt. all h\in M(I) and decreasing lattice sizes
        zs, Ms = multiCR1L_simple_Idec_waawM(index_set_I)
        # should be not worse than case 3 in average due to waawM 
    elif algno == 41:
        # constructs multiCR1L with aliasing wrt. all h\in M(I) and decreasing lattice sizes
        zs, Ms = multiCR1L_improved_Idec_waawM(index_set_I)
        # should be not worse than case 4 in average due to waawM 
    else:
        raise ValueError('Algorithm for CR1L search not chosen.')

    if 'reco_infos' not in locals() and np.prod(np.shape(MI)) == 0:
        MI, MI_index_I = build_mirrored_index_set(index_set_I)
        reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}
    elif 'reco_infos' not in locals():
        reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}

    reco_infos['kz_mod_M'] = []
    for j in range(len(Ms)):
        reco_infos['kz_mod_M'].append(np.mod(np.dot(reco_infos['M_I'], zs[j, :]), Ms[j]).astype(int))

    reco_infos.pop('M_I', None)
    return zs, Ms, reco_infos

def multiCR1L_simple_random_draw(I):
    c = 2
    log_delta = -np.log(I.shape[0])

    sizeMI = np.sum(np.prod(2 ** (I != 0), axis=1))
    N = np.max(I)
    M = nextprime(max(c * (sizeMI - 1), 2 * N)-0.5)
    L = int(np.ceil((c / (c - 1)) ** 2 * (np.log(I.shape[0]) - log_delta) / 2))

    zsret = np.random.randint(M, size=(L, I.shape[1])) - 1
    Msret = np.ones((L, 1)) * M

    return zsret, Msret

def multiCR1L_simple_greedy(I, MI, MI_index_I):
    if np.prod(np.shape(MI)) == 0:
        MI, MI_index_I = build_mirrored_index_set(I)
    reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}

    c = 2
    log_delta = -np.log(I.shape[0])
    N = np.max(I)
    M = nextprime(max(c * (MI.shape[0] - 1), 2 * N)-0.5)
    L = int(np.ceil((c / (c - 1)) ** 2 * (np.log(I.shape[0]) - log_delta) / 2))

    zs = np.random.randint(M, size=(L, MI.shape[1])) - 1
    IMarker_mirror = np.zeros((I.shape[0], zs.shape[0]), dtype=bool)

    for j in range(zs.shape[0]):
        tmp = np.mod(np.dot(MI, zs[j, :]), M)
        tmp2 = np.sort(tmp)
        tmp3 = tmp2[1:] - tmp2[:-1]
        ind4 = []

        if tmp3[0] != 0:
            ind4.append(0)
        if tmp3[-1] != 0:
            ind4.append(len(tmp2) - 1)
        tmp4 = tmp3[:-1] * tmp3[1:]
        ind4.extend(np.where(tmp4 != 0)[0] + 1)
        ind5 = np.argsort(tmp)[ind4]
        IMarker_mirror[MI_index_I[ind5], j] = 1

    Msret = []
    zsret = []
    while IMarker_mirror.shape[0] > 0:
        nImirror = np.sum(IMarker_mirror, axis=0)
        if np.sum(nImirror) == 0:
            print('Warning: multiCR1L might not be a reconstructing one')
        ind = np.argsort(nImirror)[::-1]

        Msret.append(M)
        zsret.append(zs[ind[0], :])
        ind2 = np.where(IMarker_mirror[:, ind[0]] == 1)[0]
        IMarker_mirror = np.delete(IMarker_mirror, ind2, axis=0)

    return np.array(zsret), np.array(Msret), reco_infos

def multiCR1L_simple_Idec(I, L, MI, MI_index_I):
    reco_infos = {}
    if MI is None or np.prod(np.shape(MI)) == 0:
        MI, MI_index_I = build_mirrored_index_set(I)
        reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}
    else:
        reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}

    N = np.max(I)
    M = nextprime(max(2 * (MI.shape[0] - 1), 2 * N)-0.5)

    if L is None:
        log_delta = -np.log(I.shape[0])
        L = max(10, 2 * int(np.ceil(2 * (np.log(I.shape[0]) - log_delta))))

    zs = np.random.randint(M, size=(L, MI.shape[1])) - 1
    IMarker_mirror = np.zeros((I.shape[0], zs.shape[0]), dtype=bool)

    for j in range(zs.shape[0]):
        tmp = np.mod(np.dot(MI, zs[j, :]), M)
        tmp2 = np.sort(tmp)
        tmp3 = tmp2[1:] - tmp2[:-1]
        ind4 = []

        if tmp3[0] != 0:
            ind4.append(0)
        if tmp3[-1] != 0:
            ind4.append(len(tmp2) - 1)
        tmp4 = tmp3[:-1] * tmp3[1:]
        ind4.extend(np.where(tmp4 != 0)[0] + 1)
        ind5 = np.argsort(tmp)[ind4]
        IMarker_mirror[MI_index_I[ind5], j] = 1

    nImirror = np.sum(IMarker_mirror, axis=0)
    ind = np.argsort(nImirror)[::-1]

    I_remaining_inds = np.where(IMarker_mirror[:, ind[0]] == False)[0]
    I_rem = I[I_remaining_inds, :]

    if I_rem.shape[0] > 0:
        zstmp, Mstmp = multiCR1L_simple_Idec(I_rem, L)
        Msret = np.vstack([M, Mstmp])
        zsret = np.vstack([zs[ind[0], :], zstmp])
    else:
        Msret = M
        zsret = zs[ind[0], :]

    return zsret, Msret, reco_infos

def multiCR1L_improved_Idec(I, L, MI=None, MI_index_I=None):
    reco_infos = {}
    if MI is None or np.prod(np.shape(MI)) == 0:
        MI, MI_index_I = build_mirrored_index_set(I)
        reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}
    else:
        reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}

    c = 2
    N = np.max(I)
    Mtmp = nextprime(max(2 * (MI.shape[0] - 1), max(2 * N + 1,3))-0.5)

    if L is None:
        log_delta = -np.log(I.shape[0])
        L = max(10, 2 * int(np.ceil(2 * (np.log(I.shape[0]) - log_delta))))

    nImirror = []
    IMarker_mirror = np.zeros((0, 0), dtype=bool)

    p = np.array(list(primerange(3,Mtmp+1)))  # only odd primes
    startflag = 1
    zaehler = 0  # to avoid infinite loop

    I_remaining_inds = np.arange(I.shape[0])
    M = []
    z = []

    if len(p) == 1:
        p = np.array([3, p[0]])

    while isinstance(p, np.ndarray) and len(p) > 1 and zaehler < 10:
        if startflag == 0:
            Mtmp = p[int(np.ceil(len(p) / 2))]
        else:
            zaehler += 1

        zs1 = np.random.randint(Mtmp, size=(L, MI.shape[1])) - 1
        IMarker_mirror_tmp = np.zeros((I.shape[0], zs1.shape[0]), dtype=bool)

        for j in range(zs1.shape[0]):
            tmp = np.mod(np.dot(MI, zs1[j, :]), Mtmp)
            tmp2 = np.sort(tmp)
            ind2 = np.argsort(tmp)
            tmp3 = np.diff(tmp2)
            ind4 = []
        
            if tmp3[0] != 0:
                ind4.append(0)
            if tmp3[-1] != 0:
                ind4.append(len(tmp2) - 1)
        
            tmp4 = tmp3[:-1] * tmp3[1:]
            ind4.extend(np.where(tmp4 != 0)[0] + 1)
            ind5 = ind2[ind4]
            IMarker_mirror_tmp[MI_index_I[ind5]-1, j] = 1

        # for j in range(zs1.shape[0]):
        #    tmp = np.mod(np.dot(MI, zs1[j, :]), Mtmp)
        #    tmp = tmp.astype(int)
        #    tmpnew1 = np.bincount(tmp + 1, weights=MI_index_I, minlength=Mtmp + 1).astype(int)
        #    tmpnew2 = np.bincount(tmp + 1, weights=MI_index_I, minlength=Mtmp + 1).astype(int)
        #    ind6 = np.where(tmpnew2 != 0)[0]
        #    tmpnew3 = tmpnew1[ind6]
        #    ind7 = np.where(tmpnew3 - tmpnew2[ind6] == 0)[0]
        #    ind8 = tmpnew3[ind7]
        #    IMarker_mirror_tmp[MI_index_I[ind8]-1, j] = 1

        nImirror_tmp = np.sum(IMarker_mirror_tmp, axis=0)
        max_nImirror_tmp = np.max(nImirror_tmp)
        ind = np.argmax(nImirror_tmp)

        if max_nImirror_tmp >= I.shape[0] / 2:
            I_remaining_inds = np.where(IMarker_mirror_tmp[:, ind] == False)[0]
            z = zs1[ind, :]
            M = Mtmp
            p = p[:int(np.ceil(len(p) / 2))]
            startflag = 0
        elif zaehler == 10:
            I_remaining_inds = np.where(IMarker_mirror_tmp[:, ind] == False)[0]
            z = zs1[ind, :]
            M = Mtmp
        else:
            if len(p) > 2:
                p = p[int(np.ceil(len(p) / 2)):]
            else:
                p = p[-1]

    if startflag == 1:
        print('Warning: not each of the lattices reconstructs at least half of remaining cheb_freqs')

    I_rem = I[I_remaining_inds, :]

    if I_rem.shape[0] > 0:
        zstmp, Mstmp, _ = multiCR1L_improved_Idec(I_rem, L)
        Msret = np.insert(Mstmp, 0, M)
        zsret = np.vstack([z, zstmp])
    else:
        Msret = np.array([M])
        zsret = z

    if zsret.ndim ==1:
        zsret = np.expand_dims(zsret,0)
    return zsret, Msret, reco_infos

def multiCR1L_simple_greedy_waawM(I):
    MI, MI_index_I = build_mirrored_index_set(I)
    reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}
    c = 2
    log_delta = -np.log(I.shape[0])
    N = np.max(I)
    M = nextprime(max(c * (MI.shape[0] - 1), 2 * N)-0.5)
    L = int(np.ceil((c / (c - 1)) ** 2 * (np.log(I.shape[0]) - log_delta) / 2))

    zs = np.random.randint(M, size=(L, MI.shape[1])) - 1
    IMarker_mirror1 = np.zeros((I.shape[0], zs.shape[0]), dtype=bool)

    for j in range(zs.shape[0]):
        tmp = np.mod(np.dot(MI, zs[j, :]), M)
        tmpnew1 = np.bincount(tmp + 1, weights=MI_index_I, minlength=M + 1)
        tmpnew2 = np.bincount(tmp + 1, weights=MI_index_I, minlength=M + 1)
        ind6 = np.where(tmpnew2 != 0)[0]
        tmpnew3 = tmpnew1[ind6]
        ind7 = np.where(tmpnew3 - tmpnew2[ind6] == 0)[0]
        ind8 = tmpnew3[ind7]
        IMarker_mirror1[MI_index_I[ind8], j] = 1

    Msret = []
    zsret = []
    while IMarker_mirror1.shape[0] > 0:
        nImirror = np.sum(IMarker_mirror1, axis=0)
        if np.sum(nImirror) == 0:
            print('Warning: multiCR1L might not be a reconstructing one')
        ind = np.argsort(nImirror)[::-1]

        Msret.append(M)
        zsret.append(zs[ind[0], :])
        ind2 = np.where(IMarker_mirror1[:, ind[0]] == 1)[0]
        IMarker_mirror1 = np.delete(IMarker_mirror1, ind2, axis=0)

    return np.array(zsret), np.array(Msret), reco_infos

def multiCR1L_simple_Idec_waawM(I, L):
    MI, MI_index_I = build_mirrored_index_set(I)
    reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}

    N = np.max(I)
    M = nextprime(max(2 * (MI.shape[0] - 1), 2 * N)-0.5)

    if L is None:
        log_delta = -np.log(I.shape[0])
        L = max(10, 2 * int(np.ceil(2 * (np.log(I.shape[0]) - log_delta))))

    zs = np.random.randint(M, size=(L, MI.shape[1])) - 1
    IMarker_mirror1 = np.zeros((I.shape[0], zs.shape[0]), dtype=bool)

    for j in range(zs.shape[0]):
        tmp = np.mod(np.dot(MI, zs[j, :]), M)
        tmpnew1 = np.bincount(tmp + 1, weights=MI_index_I, minlength=M + 1)
        tmpnew2 = np.bincount(tmp + 1, weights=MI_index_I, minlength=M + 1)
        ind6 = np.where(tmpnew2 != 0)[0]
        tmpnew3 = tmpnew1[ind6]
        ind7 = np.where(tmpnew3 - tmpnew2[ind6] == 0)[0]
        ind8 = tmpnew3[ind7]
        IMarker_mirror1[MI_index_I[ind8], j] = 1

    nImirror = np.sum(IMarker_mirror1, axis=0)
    ind = np.argsort(nImirror)[::-1]

    I_remaining_inds = np.where(IMarker_mirror1[:, ind[0]] == False)[0]
    I_rem = I[I_remaining_inds, :]

    if I_rem.shape[0] > 0:
        zstmp, Mstmp = multiCR1L_simple_Idec_waawM(I_rem, L)
        Msret = np.vstack([M, Mstmp])
        zsret = np.vstack([zs[ind[0], :], zstmp])
    else:
        Msret = M
        zsret = zs[ind[0], :]

    return zsret, Msret, reco_infos

def multiCR1L_improved_Idec_waawM(I, L):
    MI, MI_index_I = build_mirrored_index_set(I)
    c = 2
    reco_infos = {'M_I_index_I': MI_index_I, 'M_I': MI}

    N = np.max(I)
    Mtmp = nextprime(max(2 * (MI.shape[0] - 1), 2 * N)-0.5)

    if L is None:
        log_delta = -np.log(I.shape[0])
        L = max(10, 2 * int(np.ceil(2 * (np.log(I.shape[0]) - log_delta))))

    p = np.array(list(primerange(3,Mtmp+1)))  # only odd primes
    startflag = 1
    zaehler = 0  # to avoid infinite loop

    I_remaining_inds = np.arange(I.shape[0])
    M = []
    z = []

    if len(p) == 1:
        p = [3, p[0]]

    while len(p) > 1 and zaehler < 10:
        if startflag == 0:
            Mtmp = p[int(np.ceil(len(p) / 2))]
        else:
            zaehler += 1

        zs1 = np.random.randint(Mtmp, size=(L, MI.shape[1])) - 1
        IMarker_mirror_tmp = np.zeros((I.shape[0], zs1.shape[0]), dtype=bool)

        for j in range(zs1.shape[0]):
            tmp = np.mod(np.dot(MI, zs1[j, :]), Mtmp)
            tmpnew1 = np.bincount(tmp + 1, weights=MI_index_I, minlength=Mtmp + 1)
            tmpnew2 = np.bincount(tmp + 1, weights=MI_index_I, minlength=Mtmp + 1)
            ind6 = np.where(tmpnew2 != 0)[0]
            tmpnew3 = tmpnew1[ind6]
            ind7 = np.where(tmpnew3 - tmpnew2[ind6] == 0)[0]
            ind8 = tmpnew3[ind7]
            IMarker_mirror_tmp[MI_index_I[ind8], j] = 1

        nImirror_tmp = np.sum(IMarker_mirror_tmp, axis=0)
        max_nImirror_tmp = np.max(nImirror_tmp)
        ind = np.argmax(nImirror_tmp)

        if max_nImirror_tmp >= I.shape[0] / 2:
            I_remaining_inds = np.where(IMarker_mirror_tmp[:, ind] == False)[0]
            z = zs1[ind, :]
            M = Mtmp
            p = p[:int(np.ceil(len(p) / 2))]
            startflag = 0
        elif zaehler == 10:
            I_remaining_inds = np.where(IMarker_mirror_tmp[:, ind] == False)[0]
            z = zs1[ind, :]
            M = Mtmp
        else:
            if len(p) > 2:
                p = p[int(np.ceil(len(p) / 2)):]
            else:
                p = p[-1]

    if startflag == 1:
        print('Warning: not each of the lattices reconstructs at least half of remaining cheb_freqs')

    I_rem = I[I_remaining_inds, :]

    if I_rem.shape[0] > 0:
        zstmp, Mstmp = multiCR1L_improved_Idec_waawM(I_rem, L)
        Msret = np.vstack([M, Mstmp])
        zsret = np.vstack([z, zstmp])
    else:
        Msret = M
        zsret = z

    return zsret, Msret, reco_infos