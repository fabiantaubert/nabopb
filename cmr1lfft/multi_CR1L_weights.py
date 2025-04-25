import numpy as np
from scipy.spatial import distance
from collections import defaultdict

def multi_CR1L_weights(I, multi_lattice, n):
    """
    multi_CR1L_weights
    
    Generate weights and sampling points for a multiple Chebyshev rank-1 lattice using cosine-transformed lattice points.
    
    Parameters
    ----------
    I : ndarray of shape (m, d)
        The index set, where `m` is the number of multi-dimensional indices and `d` is the dimensionality of the indices.
    
    multi_lattice : dict
        Dictionary containing lattice data with the following keys:
        - 'Ms': ndarray of int
            Lattice size for each lattice.
        - 'zs': ndarray of shape (num_levels, d)
            Generating vector for each lattice.
        - 'reco_infos': dict, optional
            Precomputed reconstruction information. If not provided, it will be computed.
    
    n : int
        Number of samples to generate.
    
    Returns
    -------
    multi_lattice : dict
        Updated dictionary containing the subsampled lattice points and weights. Keys include:
        - 'subsampling_data': dict
            A dictionary with the following keys:
            - 'xs': ndarray
                Subsampled lattice points.
            - 'ws': ndarray
                Weights associated with the subsampled points.
            - 'js': ndarray
                Indices of the subsampled points in the original lattice.
    
    Raises
    ------
    ValueError
        If the input lattice or sampling size is too large, or if there is a mismatch in the mirrored index set.
    """

    Ms = multi_lattice['Ms']
    zs = multi_lattice['zs']
    reco_infos = {}
    
    if 'reco_infos' in multi_lattice:
        reco_infos = multi_lattice['reco_infos']
    else:
        MI, MI_index_I = build_mirrored_index_set(I)
        reco_infos['M_I_index_I'] = MI_index_I
        reco_infos['kz_mod_M'] = []
        for j in range(len(Ms)):
            reco_infos['kz_mod_M'].append(np.mod(np.dot(MI, zs[j]), Ms[j]))
        reco_infos['cur_j'] = np.nan
        multi_lattice['reco_infos'] = reco_infos
    
    num_samples = np.sum((np.array(Ms) - 1) // 2) + 1
    
    warning_flag = False
    
    if n >= num_samples:
        # print("Warning: The given full cosine-transformed multiple rank-1 lattice provides sampling scheme as desired.")
        warning_flag = True
    
    if I.shape[0] <= 15000 and np.sum(Ms) <= 10000000:
        Xi = np.zeros((np.sum((np.array(Ms) - 1) // 2) + 1, zs.shape[1]))
        Xi[:(Ms[0] + 1) // 2, :] = np.cos(2 * np.pi * np.mod(np.outer(np.arange((Ms[0] - 1) // 2 + 1), zs[0]), Ms[0]) / Ms[0])
        zaehler = (Ms[0] + 1) // 2
        for j in range(1, len(Ms)):
            Xi[zaehler:zaehler + (Ms[j] - 1) // 2, :] = np.cos(2 * np.pi * np.mod(np.outer(np.arange(1, (Ms[j] - 1) // 2 + 1), zs[j]), Ms[j]) / Ms[j])
            zaehler += (Ms[j] - 1) // 2
        
        if not warning_flag:
            diagC3 = cheby_mat_trace(Xi, I)
            
            weights = np.zeros(num_samples)
            weights[0] = diagC3[0]
            for j in range(1, len(diagC3)):
                weights[j] = weights[j-1] + diagC3[j]
            weights /= np.sum(diagC3)
            
            x = np.sort(np.random.rand(n))
            
            js = np.zeros(n, dtype=int)
            zaehler = 0
            for j in range(len(x)):
                while x[j] > weights[zaehler]:
                    zaehler += 1
                js[j] = zaehler
            
            w = diagC3[js] / np.sum(diagC3)
        else:
            js = np.arange(Xi.shape[0])
            w = np.ones(Xi.shape[0]) / Xi.shape[0]
        
        t = []
        for j in range(zs.shape[0]):
            if j == 0:
                t.append(np.mod(np.outer(np.arange((Ms[j] - 1) // 2 + 1), zs[j]), Ms[j]) / Ms[j])
            else:
                t.append(np.mod(np.outer(np.arange(1, (Ms[j] - 1) // 2 + 1), zs[j]), Ms[j]) / Ms[j])
        t = np.vstack(t)
        x = t[js]
        
        xs, ia, ic = np.unique(x, axis=0, return_index=True, return_inverse=True)
        ws = np.bincount(ic, weights=1./w)
        js1 = js[ia]
        
        multi_lattice['subsampling_data'] = {
            'xs': xs,
            'ws': ws,
            'js': js1
        }
    
    else:
        raise ValueError("Input data too big")
    
    return multi_lattice

def cheby_mat_trace(x, I):
    x = np.arccos(x)
    factor = np.prod(2.**(I != 0), axis=1)
    y = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        y[i] = np.sum(factor * np.prod(np.cos(I * x[i]), axis=1)**2)
    
    return y

