import math
import torch
from pykeops.torch import Vi, Vj
from .PD_algorithms import restart_pd_rLASSO


def SR_Lasso_Fourier(
    samples,
    values,
    frequencies,
    restarts=13,
    tol=1e-5,
    beta=2.5,
    alpha=0.1,
    lam_est=0.1,
):
    N = samples.size(0)  # number of sample points

    A_op = aNUDFT(samples, frequencies)
    AT_op = NUDFT(samples, frequencies)

    def A(x):
        return A_op(x) / math.sqrt(N)

    def AT(y):
        return AT_op(y) / math.sqrt(N)

    norm_f = torch.linalg.norm(values) / math.sqrt(len(values))
    b = values / (math.sqrt(N) * norm_f)

    lam_est = lam_est / math.sqrt(N)
    s0 = torch.zeros_like(AT(b), requires_grad=False)

    # run prim dual algorithm with restarts
    rec, vals, vals_dual = restart_pd_rLASSO(
        s0, A, AT, b, lam=lam_est, tol=tol, restarts=restarts, beta=beta, alpha=alpha
    )

    print(torch.linalg.norm(A(rec) - b))

    rec = rec * norm_f

    rec_complex = rec[:, 0] + 1j * rec[:, 1]
    rec_complex = rec_complex.unsqueeze(1)

    return rec_complex, vals, vals_dual


def SR_Lasso_Chebyshev(
    samples,
    values,
    indices,
    tol=1e-8,
    restarts=13,
    beta=2.0,
    alpha=1.0,
    lam_est=0.1,
):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype

    # precompute normalization coefficients for computational efficiency
    norm_coeffs = math.sqrt(2) ** (indices.clamp(0, 1).sum(-1, keepdim=True)).to(
        dtype=dtype
    )

    A_op = Tchebychev_eval(samples, indices, norm_coeffs, D)
    AT_op = aTchebychev_eval(samples, indices, norm_coeffs, D)

    def A(x):
        return A_op(x) / math.sqrt(N)

    def AT(y):
        return AT_op(y) / math.sqrt(N)

    norm_f = torch.linalg.norm(values) / math.sqrt(len(values))
    b = values / (math.sqrt(N) * norm_f)

    lam_est = lam_est / math.sqrt(N)
    s0 = torch.zeros_like(AT(b), requires_grad=False)

    # run prim dual algorithm with restarts
    rec, vals, vals_dual = restart_pd_rLASSO(
        s0, A, AT, b, lam=lam_est, tol=tol, restarts=restarts, beta=beta, alpha=alpha
    )

    def L2_norm_train(rec):
        return torch.linalg.norm(A_op(rec) - values / norm_f) / (math.sqrt(N))

    print("Final residual in algorithm is", L2_norm_train(rec).item())
    print("Duality Gap is", vals[-1] - vals_dual[-1])

    rec = rec * norm_f

    return rec, vals, vals_dual


def NUDFT(p, f):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, f : tensors of type torch.Tensor and shapes (N,D), (M,D)
    f_i = Vi(f)  # (M, 1, D) LazyTensor
    p_j = Vj(p)  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)  # (1, N, 1) LazyTensor
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) * x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def aNUDFT(p, f):
    # Adjoint Non-Uniform Discrete Fourier Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,2), real-valued
    # p, f : tensors of type torch.Tensor and shapes (N,D), (M,D)
    f_j = Vj(f)  # (1, M, D) LazyTensor
    p_i = Vi(p)  # (N, 1, D) LazyTensor
    x_j = Vj(0, 2)  # (1, M, 2) LazyTensor
    return ((f_j | p_i).unary("ComplexExp1j", dimres=2) | x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def aTchebychev_eval(p, k, pre, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,1), real valued
    #   p, k : tensors of type torch.Tensor and shapes (N,D)
    k_i = Vi(k)  # (M, 1, D) LazyTensor
    pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(torch.acos(p))  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def Tchebychev_eval(p, k, pre, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,2), real-valued
    #   p, f : tensors of type torch.Tensor and shapes (N,D)
    k_j = Vj(k)  # (M, 1, D) LazyTensor
    pre_j = Vj(pre)  # (M, 1, 1) LazyTensor
    p_acos_i = Vi(torch.acos(p))  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)

    tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()
    return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)
