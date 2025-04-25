import numpy as np
from sympy import nextprime, isprime, primerange

def construct_mr1l(arg, algno):
    """
    Constructs a multiple rank-1 lattice (MR1L).
    
    Parameters
    ----------
    arg : dict
        A dictionary containing the following fields:
        - `I` (array-like): Frequency index set (not required for Algorithm 6).
        - `c` (float): Oversampling factor, must be greater than 1.
        - `delta` (float): Failure probability bound, must be in the range (0, 1) (required for algorithms 2-6).
        - `n` (int): Upper bound on the number of distinct lattice sizes (used in Algorithm 1).
        - `T` (int): Upper bound on the cardinality of the frequency set `I` (used in Algorithm 6).
        - `N` (int): Refinement parameter for Algorithm 6, defining expansion of frequency set `I`.
        - `d` (int): Dimension of the frequency set `I` (used in Algorithm 6).
        - `type` (str): Algorithm 6 mode; either `'smallestprime'` or `'distinct prime'`.
        - `nrv` (int, optional): Number of random test vectors (used in Algorithm 7).
    
    algno : int
        Algorithm number specifying the construction method:
        - `1`: Randomized approach utilizing properties of `I`.
        - `2`: High-probability reconstruction with distinct prime lattice sizes.
        - `3`: Iterative high-probability reconstruction.
        - `4`: Similar to Algorithm 3 with distinct prime lattice sizes.
        - `5`: Reduces index set `I` iteratively for high-probability reconstruction.
        - `6`: Randomized approach using either smallest prime (`type='smallestprime'`) or distinct primes (`type='distinct prime'`).
        - `7`: Randomized reconstruction approach.
    
    Returns
    -------
    dict
        A dictionary containing:
        - `Ms` (list): Vector of lattice sizes.
        - `zs` (list): Array of generating vectors.
        - `mark` (int): Marker indicating the algorithm used (1 for successful reconstruction, 0 otherwise).
    
    Raises
    ------
    ValueError
        If input arguments are invalid or unsupported algorithm is selected.
    """
    
    arg['I'] = np.array(arg['I'], dtype=float)

    if check_arg(arg, algno):
        if algno == 1:
            Ms, zs = construct_mr1l_1(arg['I'], arg['c'], arg['n'])
            lattice = {'mark': 1, 'Ms': Ms, 'zs': zs}
        elif algno == 2:
            Ms, zs = construct_mr1l_2(arg['I'], arg['c'], arg['delta'])
            lattice = {'mark': 1, 'Ms': Ms, 'zs': zs}
        elif algno == 3:
            Ms, zs = construct_mr1l_3(arg['I'], arg['c'], arg['delta'])
            lattice = {'mark': 0, 'Ms': Ms, 'zs': zs}
        elif algno == 4:
            Ms, zs = construct_mr1l_4(arg['I'], arg['c'], arg['delta'])
            lattice = {'mark': 0, 'Ms': Ms, 'zs': zs}
        elif algno == 5:
            Ms, zs = construct_mr1l_5(arg['I'], arg['c'], arg['delta'], arg['C'])
            lattice = {'mark': 0, 'Ms': Ms, 'zs': zs}
        elif algno == 6:
            Ms, zs = construct_mr1l_uniform(arg['T'], arg['d'], arg['N'], arg['delta'], arg['c'], arg['type'])
            lattice = {'mark': 1, 'Ms': Ms, 'zs': zs}
        elif algno == 7:
            if 'nrv' in arg:
                Ms, zs = multi_lattice_rand(arg['I'], arg['c'], arg['nrv'])
            else:
                Ms, zs = multi_lattice_rand(arg['I'], arg['c'])
            lattice = {'mark': 1, 'Ms': Ms, 'zs': zs}
        return lattice

def check_arg(arg, algno):
    a = 1
    if algno == 1:
        if 'c' not in arg or arg['c'] <= 1:
            raise ValueError('You need to set arg.c>1')
        if 'n' not in arg or not isinstance(arg['n'], int) or arg['n'] < 1:
            raise ValueError('You need to set an integer arg.n>=1')
    elif algno == 2:
        if 'c' not in arg or arg['c'] <= 1:
            raise ValueError('You need to set arg.c>1')
        if 'delta' not in arg or arg['delta'] <= 0 or arg['delta'] >= 1:
            raise ValueError('You need to set a real value arg.delta \in (0,1)')
    elif algno == 3:
        if 'c' not in arg or arg['c'] <= 1:
            raise ValueError('You need to set arg.c>1')
        if 'delta' not in arg or arg['delta'] <= 0 or arg['delta'] >= 1:
            raise ValueError('You need to set a real value arg.delta \in (0,1)')
    elif algno == 4:
        if 'c' not in arg or arg['c'] <= 1:
            raise ValueError('You need to set arg.c>1')
        if 'delta' not in arg or arg['delta'] <= 0 or arg['delta'] >= 1:
            raise ValueError('You need to set a real value arg.delta \in (0,1)')
    elif algno == 5:
        if 'c' not in arg or arg['c'] <= 1:
            raise ValueError('You need to set arg.c>1')
        if 'delta' not in arg or arg['delta'] <= 0 or arg['delta'] >= 1:
            raise ValueError('You need to set a real value arg.delta \in (0,1)')
        if 'C' not in arg or arg['C'] < 1:
            raise ValueError('You need to set a real value arg.C \in [1,\infty)')
    elif algno == 6:
        if arg['type'] in ['smallestprime', 'distinct prime']:
            if 'T' not in arg or arg['T'] < 1 or not np.isclose(np.round(arg['T']), arg['T']):
                raise ValueError('The frequency set size T has to be larger than 1 and integer.')
            elif 'd' not in arg or not np.isclose(np.round(arg['d']), arg['d']) or arg['d'] < 1:
                raise ValueError('The frequency set has to be of integer dimension d at least 1.')
            elif 'N' not in arg:
                raise ValueError('The refinement parameter N for the frequencies has to be set to an integer, e.g. 2^k for some integer k>0.')
            elif (arg['N'] + 1) ** arg['d'] < arg['T']:
                raise ValueError('Within N^d there are not T frequencies.')
            elif 'delta' not in arg or arg['delta'] <= 0 or arg['delta'] >= 1:
                raise ValueError('The upper bound delta on the failure probability has to be larger than zero and at most one.')
            elif 'c' not in arg or arg['c'] <= 1:
                raise ValueError('The oversampling factor c has to be larger than one.')
        else:
            raise ValueError(f'The Algorithm type {arg["type"]} is unknown')
    elif algno == 7:
        if 'c' not in arg or arg['c'] <= 1:
            raise ValueError('The oversampling factor c has to be set to a value greater than one.')
        if 'I' not in arg:
            raise ValueError('You have to enter the frequency index set I for this algorithm.')
    else:
        raise ValueError(f'The Algorithm no. {algno} is unknown')
    return a

def construct_mr1l_1(I, c, n):
    T = I.shape[0]
    lambda_ = c * (T - 1)
    N_I = np.max(np.max(I, axis=0) - np.min(I, axis=0))
    p0 = max(lambda_, N_I, 3275)
    pn_ub = (1 + 1 / (2 * (np.log(p0)) ** 2)) ** n * p0
    prime_cands = np.array([p for p in range(2, int(pn_ub)+1) if isprime(p)])
    prime_cands = prime_cands[prime_cands > lambda_]
    PIlambda_n = get_PIlambda_n(I, prime_cands, n)
    ell = 0
    Ms = []
    zs = []
    tilde_I = np.zeros((0, I.shape[1]))
    while I.shape[0] > tilde_I.shape[0]:
        ell += 1
        Ms.append(PIlambda_n[np.random.randint(n)+1])
        zs.append(np.random.randint(Ms[-1], size=I.shape[1]))
        tmp = np.mod(I @ zs[-1], Ms[-1])
        tmpsort = np.sort(tmp)
        tmpsortdiff = tmpsort[1:] - tmpsort[:-1]
        ind2 = []
        if tmpsortdiff[0] != 0:
            ind2.append(0)
        if tmpsortdiff[-1] != 0:
            ind2.append(len(tmpsort) - 1)
        ind2tmp = np.where(tmpsortdiff[:-1] * tmpsortdiff[1:] != 0)[0]
        ind2.extend(ind2tmp + 1)
        stI = tilde_I.shape[0]
        tilde_I = np.unique(np.vstack((tilde_I, I[tmpsort[ind2], :])), axis=0)
        if stI == tilde_I.shape[0]:
            Ms.pop()
            zs.pop()
            ell -= 1
    return Ms, zs

def construct_mr1l_2(I, c, delta):
    T = I.shape[0]
    N_I = np.max(np.max(I) - np.min(I))
    s = np.ceil((c / (c - 1)) ** 2 * (np.log(T) - np.log(delta)) / 2)
    lambda_ = c * (T - 1)
    p0 = max(lambda_, 3275, N_I)
    pn_ub = (1 + 1 / (2 * (np.log(p0)) ** 2)) ** s * p0
    prime_cands = np.array([p for p in range(2, int(pn_ub)) if isprime(p)])
    prime_cands = prime_cands[prime_cands > lambda_]
    PIlambda_n = np.sort(get_PIlambda_n(I, prime_cands, s))
    ell = 0
    Ms = []
    zs = []
    tilde_I = np.zeros((0, I.shape[1]))
    while I.shape[0] > tilde_I.shape[0]:
        ell += 1
        Ms.append(PIlambda_n[ell - 1])
        zs.append(np.random.randint(Ms[-1], size=I.shape[1]) - 1)
        tmp = np.mod(I @ zs[-1], Ms[-1])
        tmpsort = np.sort(tmp)
        tmpsortdiff = tmpsort[1:] - tmpsort[:-1]
        ind2 = []
        if tmpsortdiff[0] != 0:
            ind2.append(0)
        if tmpsortdiff[-1] != 0:
            ind2.append(len(tmpsort) - 1)
        ind2tmp = np.where(tmpsortdiff[:-1] * tmpsortdiff[1:] != 0)[0]
        ind2.extend(ind2tmp + 1)
        stI = tilde_I.shape[0]
        tilde_I = np.unique(np.vstack((tilde_I, I[tmpsort[ind2], :])), axis=0)
        if stI == tilde_I.shape[0]:
            Ms.pop()
            zs.pop()
            ell -= 1
        if ell == s:
            if tilde_I.shape[0] != I.shape[0]:
                print('Algorithm 2 did not determine an reco mr1l.')
            break
    return Ms, zs

def construct_mr1l_3(I, c, delta):
    T1 = I.shape[0]
    ell = 0
    Ms = []
    zs = []
    while I.shape[0] > 0:
        ell += 1
        T = I.shape[0]
        s = int(np.ceil((c / (c - 1)) ** 2 * (np.log(T) + np.log(T1) - np.log(delta)) / 2))
        lambda_ = c * (T - 1)
        if len(Ms) < ell:
            Ms.append(get_PIlambda_1(I, lambda_))
        else:
            Ms[ell-1] = get_PIlambda_1(I, lambda_)
        v = np.random.randint(Ms[-1], size=(s, I.shape[1]))
        aktind = np.arange(I.shape[0])
        aktj = -1
        for j in range(s):
            tmp = np.mod(I @ v[j, :], Ms[-1])
            ind1 = np.argsort(tmp)
           # tmpsort = np.sort(tmp)
            tmpsort = tmp[ind1]
            tmpsortdiff = tmpsort[1:] - tmpsort[:-1]
            ind2 = np.where(tmpsortdiff == 0)[0]
            ind2 = np.unique(np.concatenate((ind2, ind2 + 1)))
            if len(aktind) > len(ind2):
                aktind = ind1[ind2]
                aktj = j
        if aktj != -1:
            I = I[aktind, :]
            zs.append(v[aktj, :])
        else:
            ell -= 1
            print('Nothing done in Algorithm 3.')
    return Ms, zs

def construct_mr1l_4(I, c, delta):
    T1 = I.shape[0]
    ell = 0
    Ms = []
    zs = []
    while I.shape[0] > 0:
        ell += 1
        T = I.shape[0]
        s = np.ceil((c / (c - 1)) ** 2 * (np.log(T) + np.log(T1) - np.log(delta)) / 2)
        lambda_ = c * (T - 1)
        N_I = np.max(np.max(I) - np.min(I))
        p0 = max(np.array([p for p in range(2, int(max(lambda_, 3275, N_I))) if isprime(p)]))
        p1 = np.array([p for p in range(2, int((1 + 1 / (2 * (np.log(p0)) ** 2)) ** ell * p0)) if isprime(p)])
        p1 = p1[p1 > lambda_]
        p1 = get_PIlambda_n(I, p1, ell)
        if ell > 1:
            p1 = np.setdiff1d(p1, Ms[:ell - 1])
        Ms.append(np.min(p1))
        v = np.random.randint(Ms[-1], size=(s, I.shape[1])) - 1
        aktind = np.arange(I.shape[0])
        aktj = 0
        for j in range(s):
            tmp = np.mod(I @ v[j, :], Ms[-1])
            tmpsort = np.sort(tmp)
            tmpsortdiff = tmpsort[1:] - tmpsort[:-1]
            ind2 = np.where(tmpsortdiff == 0)[0]
            ind2 = np.unique(np.concatenate((ind2, ind2 + 1)))
            if len(aktind) > len(ind2):
                aktind = tmpsort[ind2]
                aktj = j
        if aktj != 0:
            I = I[aktind, :]
            zs.append(v[aktj, :])
        else:
            ell -= 1
            print('Nothing done in Algorithm 4.')
    return Ms, zs

def construct_mr1l_5(I, c, _, C):
    l = 0
    T = I.shape[0]
    Ms = []
    zs = []
    ell = 0
    while I.shape[0] > 0:
        Mtmp, ImodM, IRest = get_MICc(I, C, c)
        tilde_ImodM = np.zeros((0, ImodM.shape[1]))
        while ImodM.shape[0] > tilde_ImodM.shape[0]:
            ell += 1
            Ms.append(Mtmp)
            zs.append(np.random.randint(Ms[-1], size=ImodM.shape[1]) - 1)
            tmp = np.mod(ImodM @ zs[-1], Ms[-1])
            tmpsort = np.sort(tmp)
            tmpsortdiff = tmpsort[1:] - tmpsort[:-1]
            ind2 = []
            if tmpsortdiff[0] != 0:
                ind2.append(0)
            if tmpsortdiff[-1] != 0:
                ind2.append(len(tmpsort) - 1)
            ind2tmp = np.where(tmpsortdiff[:-1] * tmpsortdiff[1:] != 0)[0]
            ind2.extend(ind2tmp + 1)
            stImodM = tilde_ImodM.shape[0]
            tilde_ImodM = np.unique(np.vstack((tilde_ImodM, ImodM[tmpsort[ind2], :])), axis=0)
            if stImodM == tilde_ImodM.shape[0]:
                Ms.pop()
                zs.pop()
                ell -= 1
        I = IRest
    return Ms, zs

def construct_mr1l_uniform(T, d, N, delta, c, type):
    if type == 'smallestprime':
        return construct_mr1l_uniform1(T, d, N, delta, c)
    elif type == 'distinct prime':
        return construct_mr1l_uniform2(T, d, N, delta, c)

def construct_mr1l_uniform1(T, d, N, delta, c):
    c = max(N / (T - 1), c)
    lambda_ = c * (T - 1)
    s = np.ceil((c / (c - 1)) ** 2 * (np.log(T) - np.log(delta)) / 2)
    M = np.array([p for p in range(2, int(2 * lambda_)) if isprime(p)])
    M = M[M > lambda_]
    M = np.min(M)
    zs = np.random.randint(M, size=(s, d)) - 1
    Ms = np.full(s, M)
    return Ms, zs

def construct_mr1l_uniform2(T, d, N, delta, c):
    c = max(N / (T - 1), c)
    lambda_ = c * (T - 1)
    s = np.ceil((c / (c - 1)) ** 2 * (np.log(T) - np.log(delta)) / 2)
    M = np.array([p for p in range(2, int(2 * lambda_)) if isprime(p)])
    M = M[M > lambda_]
    factor = 2
    while len(M) < s:
        factor += 1
        M = np.array([p for p in range(2, int(factor * lambda_)) if isprime(p)])
        M = M[M > lambda_]
    Ms = np.zeros(s)
    zs = np.zeros((s, d))
    for j in range(s):
        Ms[j], ind = np.min(M), np.argmin(M)
        M[ind] = np.nan
        zs[j, :] = np.random.randint(Ms[j], size=d) - 1
    return Ms, zs

def multi_lattice_rand(h, c, nrv=None):
    h1 = np.array(h, dtype=float)
    if nrv is None:
        nrv = 10 * h1.shape[1] * max(np.log(c * h1.shape[0]), 1)
    Ms = []
    zs = []
    d = h1.shape[1]
    k = 1
    hred = 1
    hredsave = []
    while h1.shape[0] > 0:
        if hred != 0:
            try:
                Mcand = nextprime(c * h1.shape[0])
            except:
                Mcand = nextprime(c * h1.shape[0] + 1)
        else:
            try:
                Mcand = nextprime(Mcand)
            except:
                Mcand = nextprime(Mcand + 1)
        hred = 0
        for j in range(nrv):
            zcand = np.random.randint(Mcand - 1, size=d)
            y = np.mod(h1 @ zcand, Mcand)
            y = np.sort(y)
            y = np.setdiff1d(y, y[np.where(y[1:] - y[:-1] == 0)[0]])
            if len(y) > hred:
                hred = len(y)
                Ms.append(Mcand)
                zs.append(zcand)
        if hred > 0:
            y = np.mod(h1 @ zs[-1], Ms[-1])
            y, ind = np.sort(y), np.argsort(y)
            _, ind2 = np.setdiff1d(y, y[np.where(y[1:] - y[:-1] == 0)[0]])
            h1 = np.setdiff1d(h1, h1[ind[ind2], :], axis=0)
            k += 1
            hredsave.append(hred)
    return Ms, zs

def get_MICc(I, C, c):
    N_I = np.max(np.max(I) - np.min(I))
    if I.shape[0] / C > N_I:
        p0 = max(3275, c * (I.shape[0] - 1))
        pn_ub = (1 + 1 / (2 * (np.log(p0)) ** 2)) * p0
        p = np.array([p for p in range(2, int(pn_ub)) if isprime(p)])
        p = p[p > c * (I.shape[0] - 1)]
        MICc = np.min(p)
        ImodM = I
        IRest = np.array([])
    else:
        p0 = max(3275, c * (I.shape[0] - 1), N_I)
        pn_ub = (1 + 1 / (2 * (np.log(p0)) ** 2)) * p0
        p = np.array([p for p in range(2, int(pn_ub)) if isprime(p)])
        ptmp = p[p > p0]
        p = p[p <= np.min(ptmp)]
        p = np.sort(p[p >= I.shape[0] / C])
        for ptest in p:
            ImodM, _, ind2 = np.unique(np.mod(I, ptest), axis=0, return_index=True, return_inverse=True)
            if ptest > c * (I.shape[0] - 1):
                sortind2 = np.sort(ind2)
                sortind2diff = sortind2[1:] - sortind2[:-1]
                ind4 = np.where(sortind2diff == 0)[0]
                ind4 = np.unique(np.concatenate((ind4, ind4 + 1)))
                IRest = I[ind2[ind4], :]
                if (1 - 1 / C) * I.shape[0] >= IRest.shape[0]:
                    MICc = ptest
                    break
    return MICc, ImodM, IRest

def get_PIlambda_1(I, lambda_):
    N_I = np.max(np.max(I, axis=0) - np.min(I, axis=0))
    prime_cands = np.array(list(primerange(2,int(2 * max(lambda_, N_I))+1)))
    #prime_cands = np.array([p for p in range(2, int(2 * max(lambda_, N_I))) if isprime(p)])
    prime_cands = prime_cands[prime_cands > lambda_]
    for j in range(len(prime_cands)):
        if prime_cands[j] > N_I or np.unique(np.mod(I, prime_cands[j]), axis=0).shape[0] == I.shape[0]:
            return prime_cands[j]

def get_PIlambda_n(I, prime_cands, n):
    prime_cands = np.sort(prime_cands)
    ps = np.zeros(n)
    zaehler = 0
    N_I = np.max(np.max(I) - np.min(I))
    for j in range(len(prime_cands)):
        if prime_cands[j] > N_I or np.unique(np.mod(I, prime_cands[j]), axis=0).shape[0] == I.shape[0]:
            ps[zaehler] = prime_cands[j]
            zaehler += 1
        if zaehler == n:
            break
    return ps

