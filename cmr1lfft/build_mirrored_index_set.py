import numpy as np

def build_mirrored_index_set(index_set_I, flag=None):
    """
    Constructs a mirrored index set by reflecting the input index set along its dimensions.
    
    Parameters
    ----------
    index_set_I : ndarray
        The input index set, where each row is an index in a d-dimensional space. 
        Must not contain negative frequencies.
    flag : str, optional
        If 'except_1d', excludes a single dimension from the mirroring process.
        By default, all dimensions are mirrored.
    
    Returns
    -------
    M_I : ndarray
        The mirrored index set, including the original indices and their reflections.
    M_I_index_I : ndarray
        An array of indices mapping each row in `M_I` back to the corresponding 
        original row in `index_set_I`.
    
    Raises
    ------
    ValueError
        If `index_set_I` contains negative frequencies or if an internal error occurs 
        during processing.
    """
    double_check = 0

    if np.min(index_set_I) < 0:
        raise ValueError('input index set must not contain frequencies less than zero')

    d = index_set_I.shape[1]

    if flag == 'except_1d':
        s = np.argmax(np.sum(index_set_I != 0, axis=0))
        ds = np.concatenate((np.arange(1, s),np.arange(s + 1, d + 1)))
    else:
        s = -1
        ds = np.arange(1, d + 1)

    num_M_I_rows = int(np.sum(2 ** (np.sum(index_set_I[:, ds-1] != 0, axis=1))))
    M_I = np.nan * np.zeros((num_M_I_rows, d))
    with np.errstate(invalid='ignore'):
        M_I_index_I = (np.nan * np.zeros((num_M_I_rows))).astype(int)
    end_row = index_set_I.shape[0]
    M_I[:end_row, :] = index_set_I
    M_I_index_I[:end_row] = np.squeeze(np.arange(1, end_row + 1).reshape(-1, 1))

    for t in range(d):
        if t == s:
            continue
        ind_t_nz = np.where(M_I[:end_row, t] != 0)[0]
        freq_new = M_I[ind_t_nz, :].copy()
        freq_new[:, t] = -M_I[ind_t_nz, t]
        new_start_row = end_row
        end_row = new_start_row + len(ind_t_nz)
        M_I[new_start_row:end_row, :] = freq_new
        M_I_index_I[new_start_row:end_row] = M_I_index_I[ind_t_nz]

    if end_row != num_M_I_rows:
        raise ValueError('internal error')

    if double_check:
        M_I_2 = index_set_I.copy()
        M_I_index_I_2 = np.arange(1, index_set_I.shape[0] + 1).reshape(-1, 1)

        for t in range(d):
            ind_t_nz = np.where(M_I_2[:, t] != 0)[0]
            freq_new = M_I_2[ind_t_nz, :].copy()
            freq_new[:, t] = -M_I_2[ind_t_nz, t]
            M_I_2 = np.vstack((M_I_2, freq_new))
            M_I_index_I_2 = np.vstack((M_I_index_I_2, M_I_index_I_2[ind_t_nz]))

        if np.max(np.abs(M_I - M_I_2)) > 0:
            raise ValueError('M_I mismatch')

        if np.max(np.abs(M_I_index_I - M_I_index_I_2)) > 0:
            raise ValueError('M_I_index_I mismatch')

    return M_I, M_I_index_I