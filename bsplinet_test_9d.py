import numpy as np

def bsplinet_test_9d(x):
    """
    Evaluate a 9-dimensional B-spline test function.

    This function computes the sum of two tensor-product B-spline basis evaluations:
    - A 4-dimensional second-order B-spline on dimensions [0, 2, 3, 6]
    - A 5-dimensional fourth-order B-spline on dimensions [1, 4, 5, 7, 8]

    Parameters
    ----------
    x : ndarray of shape (M, 9)
        Input array where each row is a 9-dimensional point at which to 
        evaluate the function.

    Returns
    -------
    valout : ndarray of shape (M,)
        Function values at each input point.
    """
    
    if x.shape[1] != 9:
        raise ValueError('input must be M x 9 matrix')
    valout = bspline2t_nd(x[:, [0, 2, 3, 6]]) + bspline4t_nd(x[:, [1, 4, 5, 7, 8]])
    return valout

def bsplinet_test_9d_chat(freq_out):
    """
    Compute Chebyshev coefficients of the 9-dimensional B-spline test function.

    Also gives the squared Chebyshev norm of the test function.

    Parameters
    ----------
    freq_out : ndarray of shape (K, 9)
        Array of 9-dimensional frequency vectors at which to evaluate 
        the Chebyshev coefficients.

    Returns
    -------
    fhat : np.ndarray, shape (N,)
        The Chebyshev coefficients of the test function at the given frequencies.

    norm_fct_square : float
        Squared norm of the function.
    """
    fhat = np.zeros((freq_out.shape[0]))

    ind = np.where(np.sum(np.abs(freq_out[:, [1, 4, 5, 7, 8]]), axis=1) == 0)[0]
    fhat[ind] = fhat[ind] + bspline2t_chat_nd(freq_out[ind,:][:,[0,2,3,6]])

    ind = np.where(np.sum(np.abs(freq_out[:, [0, 2, 3, 6]]), axis=1) == 0)[0]
    fhat[ind] = fhat[ind] + bspline4t_chat_nd(freq_out[ind,:][:,[1, 4, 5, 7, 8]]).squeeze()

    _, bspline2t_normcheb_d4_sq = bspline2t_normcheb(4)
    _, bspline4t_normcheb_d5_sq = bspline4t_normcheb(5)
    norm_fct_square = bspline2t_normcheb_d4_sq + bspline4t_normcheb_d5_sq + 2 * bspline2t_chat_nd(np.zeros((1,4))) * bspline4t_chat_nd(np.zeros((1,5)))
    return fhat, norm_fct_square.squeeze()

def bspline2t_1d(x):
    x = np.array(x).reshape(-1)

    val = np.zeros_like(x)
    iremain = np.arange(len(x))
    iremain = iremain[x >= -9/2]

    ind_pos = np.where(x[iremain] < -5/2)[0]
    x_cur = x[iremain[ind_pos]]
    val[iremain[ind_pos]] = 1/32 * (2 * x_cur + 9) ** 2
    iremain = iremain[x[iremain] >= -5/2]

    ind_pos = np.where(x[iremain] < -1/2)[0]
    x_cur = x[iremain[ind_pos]]
    val[iremain[ind_pos]] = 3/16 - 3/4 * x_cur - 1/4 * x_cur ** 2
    iremain = iremain[x[iremain] >= -1/2]

    ind_pos = np.where(x[iremain] < 3/2)[0]
    x_cur = x[iremain[ind_pos]]
    val[iremain[ind_pos]] = 9/32 - 3/8 * x_cur + 1/8 * x_cur ** 2
    return val

def bspline2t_chat_1d(k):
    k = np.array(k).reshape(-1)
    chat = (9 * np.sqrt(3) * k * np.cos((2 * k * np.pi) / 3) - 9 * (-2 + k**2) * np.sin((2 * k * np.pi) / 3)) / (8 * k * (4 - 5 * k**2 + k**4) * np.pi)
    chat[k == 2] = (9 * np.sqrt(3)) / (128 * np.pi)
    chat[k == 1] = -(1 / 2) + (9 * np.sqrt(3)) / (32 * np.pi)
    chat[k == 0] = 1 / 4 + (9 * np.sqrt(3)) / (64 * np.pi)
    return chat

def bspline2t_chat_nd(freq):
    chat = bspline2t_chat_1d(freq[:, 0])
    for t in range(1, freq.shape[1]):
        chat = chat * bspline2t_chat_1d(freq[:, t])
    return chat

def bspline2t_chat_norm1(d=1):
    val = (np.sum(np.abs(bspline2t_chat_1d(np.arange(3)))) + 3/128 * (4 - 5 * np.sqrt(3)/np.pi)) ** d
    return val

def bspline2t_nd(nodes):
    val = bspline2t_1d(nodes[:, 0])
    for t in range(1, nodes.shape[1]):
        val = val * bspline2t_1d(nodes[:, t])
    return val

def bspline2t_normcheb(d=1):
    norm_cheb = 0.7257821329550897 ** d
    norm_cheb_sq = 0.228069332267236 ** d
    return norm_cheb, norm_cheb_sq

def bspline4t_1d(x):
    x = np.array(x).reshape(-1)

    val = np.zeros_like(x)
    iremain = np.arange(len(x))
    iremain = iremain[x >= -15/2]

    ind_pos = np.where(x[iremain] < -11/2)[0]
    x_cur = x[iremain[ind_pos]]
    val[iremain[ind_pos]] = (1/6144) * (2*x_cur + 15)**4
    iremain = iremain[x[iremain] >= -11/2]

    ind_pos = np.where(x[iremain] < -7/2)[0]
    x_cur = x[iremain[ind_pos]]
    val[iremain[ind_pos]] = -5645/1536 - (205/48)*x_cur - (95/64)*x_cur**2 - (5/24)*x_cur**3 - (1/96)*x_cur**4
    iremain = iremain[x[iremain] >= -7/2]

    ind_pos = np.where(x[iremain] < -3/2)[0]
    x_cur = x[iremain[ind_pos]]
    val[iremain[ind_pos]] = 715/3072 + (25/128)*x_cur + (55/128)*x_cur**2 + (5/32)*x_cur**3 + (1/64)*x_cur**4
    iremain = iremain[x[iremain] >= -3/2]

    ind_pos = np.where(x[iremain] < 1/2)[0]
    x_cur = x[iremain[ind_pos]]
    val[iremain[ind_pos]] = 155/1536 - (5/32)*x_cur + (5/64)*x_cur**2 - (1/96)*x_cur**4
    iremain = iremain[x[iremain] >= 1/2]

    ind_pos = np.where(x[iremain] < 5/2)[0]
    x_cur = x[iremain[ind_pos]]
    val[iremain[ind_pos]] = (1/6144) * (2*x_cur - 5)**4
    return val

def bspline4t_chat_1d(k):
    k = np.array(k).reshape(-1, 1)
    chat = ((900 * np.sqrt(3) * k * (-9 + k**2) * np.cos(k * np.pi) / 3) + 90 * (152 - 75 * k**2 + 3 * k**4) * np.sin(k * np.pi / 3)) / (768 * k * (-16 + k**2) * (-9 + k**2) * (-4 + k**2) * (-1 + k**2) * np.pi)
    chat[k == 4] = -(7/9216) - (93 * np.sqrt(3)) / (114688 * np.pi)
    chat[k == 3] = (5 * (-14 + (27 * np.sqrt(3)) / np.pi)) / 32256
    chat[k == 2] = 181/4608 - (39 * np.sqrt(3)) / (4096 * np.pi)
    chat[k == 1] = -(95/576) + (33 * np.sqrt(3)) / (2048 * np.pi)
    chat[k == 0] = 2603/18432 - (75/8192) * np.sqrt(3) / np.pi
    return chat

def bspline4t_chat_nd(freq):
    chat = bspline4t_chat_1d(freq[:, 0])
    for t in range(1, freq.shape[1]):
        chat = chat * bspline4t_chat_1d(freq[:, t])
    return chat

def bspline4t_nd(nodes):
    val = bspline4t_1d(nodes[:, 0])
    for t in range(1, nodes.shape[1]):
        val = val * bspline4t_1d(nodes[:, t])
    return val

def bspline4t_normcheb(d=1):  
    norm_cheb = np.sqrt((3904915/113246208)*np.pi - (1356109/234881024)*np.sqrt(3))**d
    norm_cheb_sq = 0.0440535567777421**d
    return norm_cheb, norm_cheb_sq