import numpy as np

def bspline_test_10d(x):
    """
    Evaluate a 10-dimensional B-spline test function.

    The function is a sum of tensor-product B-splines of orders 2, 4, and 6, 
    each applied to selected dimensions of the 10d input.

    Parameters
    ----------
    x : ndarray of shape (M, 10)
        Input array where each row is a 10-dimensional point at which to 
        evaluate the function.

    Returns
    -------
    valout : ndarray of shape (M,)
        Function values at each input point.
    """
    
    if x.shape[1] != 10:
        raise ValueError('input must be M x 10 matrix')

    valout = bspline_o2(x[:, [0, 2, 7]]) + bspline_o4(x[:, [1, 4, 5, 9]]) + bspline_o6(x[:, [3, 6, 8]])
    return valout

def bspline_o2(x):
    x = x - np.floor(x)
    val = np.ones(x.shape[0])

    for t in range(x.shape[1]):
        ind = np.where((0 <= x[:, t]) & (x[:, t] < 1/2))[0]
        if ind.size > 0:
            val[ind] *= 4 * x[ind, t]

        ind = np.where((1/2 <= x[:, t]) & (x[:, t] < 1))[0]
        if ind.size > 0:
            val[ind] *= 4 * (1 - x[ind, t])

        val = np.sqrt(3/4) * val

    return val

def bspline_o4(x):
    x = x - np.floor(x)
    val = np.ones(x.shape[0])

    for t in range(x.shape[1]):
        ind = np.where((0 <= x[:, t]) & (x[:, t] < 1/4))[0]
        if ind.size > 0:
            val[ind] *= (128/3) * (x[ind, t] ** 3)

        ind = np.where((1/4 <= x[:, t]) & (x[:, t] < 2/4))[0]
        if ind.size > 0:
            val[ind] *= (8/3 - 32 * x[ind, t] + 128 * (x[ind, t] ** 2) - 128 * (x[ind, t] ** 3))

        ind = np.where((2/4 <= x[:, t]) & (x[:, t] < 3/4))[0]
        if ind.size > 0:
            val[ind] *= (-88/3 - 256 * (x[ind, t] ** 2) + 160 * x[ind, t] + 128 * (x[ind, t] ** 3))

        ind = np.where((3/4 <= x[:, t]) & (x[:, t] < 1))[0]
        if ind.size > 0:
            val[ind] *= (128/3 - 128 * x[ind, t] + 128 * (x[ind, t] ** 2) - (128/3) * (x[ind, t] ** 3))

        val = np.sqrt(315/604) * val
    return val

def bspline_o6(x):
    x = x - np.floor(x)
    val = np.ones(x.shape[0])

    for t in range(x.shape[1]):
        ind = np.where((0 <= x[:, t]) & (x[:, t] < 1/6))[0]
        if ind.size > 0:
            val[ind] *= (1944/5) * (x[ind, t] ** 5)

        ind = np.where((1/6 <= x[:, t]) & (x[:, t] < 2/6))[0]
        if ind.size > 0:
            val[ind] *= (3/10 - 9 * x[ind, t] + 108 * (x[ind, t] ** 2) - 648 * (x[ind, t] ** 3) + 1944 * (x[ind, t] ** 4) - 1944 * (x[ind, t] ** 5))

        ind = np.where((2/6 <= x[:, t]) & (x[:, t] < 3/6))[0]
        if ind.size > 0:
            val[ind] *= (-237/10 + 351 * x[ind, t] - 2052 * (x[ind, t] ** 2) + 5832 * (x[ind, t] ** 3) - 7776 * (x[ind, t] ** 4) + 3888 * (x[ind, t] ** 5))

        ind = np.where((3/6 <= x[:, t]) & (x[:, t] < 4/6))[0]
        if ind.size > 0:
            val[ind] *= (2193/10 + 7668 * (x[ind, t] ** 2) - 2079 * x[ind, t] + 11664 * (x[ind, t] ** 4) - 13608 * (x[ind, t] ** 3) - 3888 * (x[ind, t] ** 5))

        ind = np.where((4/6 <= x[:, t]) & (x[:, t] < 5/6))[0]
        if ind.size > 0:
            val[ind] *= (-5487/10 + 3681 * x[ind, t] - 9612 * (x[ind, t] ** 2) + 12312 * (x[ind, t] ** 3) - 7776 * (x[ind, t] ** 4) + 1944 * (x[ind, t] ** 5))

        ind = np.where((5/6 <= x[:, t]) & (x[:, t] < 1))[0]
        if ind.size > 0:
            val[ind] *= (1944/5 - 1944 * x[ind, t] + 3888 * (x[ind, t] ** 2) - 3888 * (x[ind, t] ** 3) + 1944 * (x[ind, t] ** 4) - (1944/5) * (x[ind, t] ** 5))

        val = np.sqrt(277200/655177) * val
    return val

def bspline_test_10d_fouriercoeff(freq_out):
    """
    Compute Fourier coefficients of the 10-dimensional B-spline test function.

    Also gives the squared norm of the test function.

    Parameters
    ----------
    freq_out : ndarray of shape (K, 10)
        Array of 10-dimensional frequency vectors at which to evaluate 
        the Fourier coefficients.

    Returns
    -------
    fhat : ndarray of shape (K,)
        Fourier coefficients at the specified frequencies.

    norm_fct_square : float
        Squared norm of the function.
    """
    
    fhat = np.zeros((freq_out.shape[0]))

    ind = np.where(np.sum(np.abs(freq_out[:, [1, 3, 4, 5, 6, 8, 9]]), axis=1) == 0)[0]
    fhat[ind] += bspline_o2_hat(freq_out[ind, :][:, [0, 2, 7]])

    ind = np.where(np.sum(np.abs(freq_out[:, [0, 2, 3, 6, 7, 8]]), axis=1) == 0)[0]
    fhat[ind] += bspline_o4_hat(freq_out[ind, :][:, [1, 4, 5, 9]])

    ind = np.where(np.sum(np.abs(freq_out[:, [0, 1, 2, 4, 5, 7, 9]]), axis=1) == 0)[0]
    fhat[ind] += bspline_o6_hat(freq_out[ind, :][:, [3, 6, 8]])

    norm_fct_square = (3 + 
                       2 * bspline_o2_hat(np.zeros((1, 3))) * 
                       bspline_o4_hat(np.zeros((1, 4))) + 
                       2 * bspline_o2_hat(np.zeros((1, 3))) * 
                       bspline_o6_hat(np.zeros((1, 3))) + 
                       2 * bspline_o4_hat(np.zeros((1, 4))) * 
                       bspline_o6_hat(np.zeros((1, 3))))
    
    return fhat, norm_fct_square

def bspline_o2_hat(k):
    val = np.ones((k.shape[0]))

    for t in range(k.shape[1]):
        ind = np.where(k[:, t] != 0)[0]
        if ind.size > 0:
            val[ind] = val[ind] * (bspline_sinc(np.pi / 2 * k[ind, t]) ** 2) * np.cos(np.pi * k[ind, t])
        val = np.sqrt(3 / 4) * val
    return val


def bspline_o4_hat(k):
    val = np.ones((k.shape[0]))

    for t in range(k.shape[1]):
        ind = np.where(k[:, t] != 0)[0]
        if ind.size > 0:
            val[ind] = val[ind] * (bspline_sinc(np.pi / 4 * k[ind, t]) ** 4) * np.cos(np.pi * k[ind, t])
        val = np.sqrt(315 / 604) * val
    return val

def bspline_o6_hat(k):
    val = np.ones((k.shape[0]))

    for t in range(k.shape[1]):
        ind = np.where(k[:, t] != 0)[0]
        if ind.size > 0:
            val[ind] = val[ind] * (bspline_sinc(np.pi / 6 * k[ind, t]) ** 6) * np.cos(np.pi * k[ind, t])
        val = np.sqrt(277200 / 655177) * val
    return val

def bspline_sinc(x):
    if x.ndim != 1:
        raise ValueError('only column vectors are supported')

    val = np.ones(x.shape)
    ind = np.where(x != 0)[0]
    val[ind] = np.sin(x[ind]) / x[ind]

    return val