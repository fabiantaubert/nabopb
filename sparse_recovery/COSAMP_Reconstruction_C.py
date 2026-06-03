import math
import torch
from pykeops.torch import Vi, Vj
from deepinv.optim.utils import least_squares


def COSAMP_Chebyshev(samples, values, indices, sparsity=5000, num_iters=100, tol=1e-7):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points

    # precompute normalization coefficients for computational efficiency
    norm_coeffs = math.sqrt(2) ** (indices.clamp(0, 1).sum(-1, keepdim=True))

    # store samples in acos format
    samples_acos = torch.acos(samples)

    normalization = torch.sqrt(
        normalization_Techebychev(samples_acos, indices, norm_coeffs, D) / N
    )

    A_handle = Tchebychev_eval(D)
    AT_handle = aTchebychev_eval(D)

    def A(x, mask_indices=None):
        p = samples_acos
        if mask_indices == None:
            f = indices
            pre = norm_coeffs
        else:
            f = indices[mask_indices]
            pre = norm_coeffs[mask_indices]
        return A_handle(x, p, f, pre) / math.sqrt(N)

    def AT(x, mask_indices=None):
        p = samples_acos
        if mask_indices == None:
            f = indices
            pre = norm_coeffs
        else:
            f = indices[mask_indices]
            pre = norm_coeffs[mask_indices]
        return AT_handle(x, p, f, pre) / math.sqrt(N)

    b = values / math.sqrt(N)

    norm = torch.linalg.norm(b)
    # run COSAMP reconstruction
    rec = COSAMP(
        A,
        AT,
        normalization,
        b,
        indices,
        samples_acos,
        sparsity=sparsity,
        num_iters=num_iters,
        tol=tol * norm,
    )
    residuals = torch.linalg.norm(A(rec) - b)
    print("Final residual:", residuals.item())
    return rec, residuals


def COSAMP(A, AT, normalization, b, f, p, sparsity=5000, num_iters=100, tol=1e-3):
    device = p.device
    dtype = p.dtype
    sparsity = min(sparsity, f.size(0))
    two_sparsity = min(2 * sparsity, f.size(0))

    x = torch.zeros(normalization.numel(), device=device, dtype=dtype)
    z_2 = torch.zeros_like(x)
    support = torch.empty(0, device=device, dtype=torch.long)
    inv_diag = 1.0 / (normalization.flatten() + 1e-10)

    res = b.clone()
    res_norm_old = 0
    Aop = MaskedOperator(A, AT)

    for j in range(num_iters):
        torch.abs(AT(res).flatten() * inv_diag, out=z_2)
        _, ind = torch.topk(z_2, k=two_sparsity)

        support_cand = torch.cat((support, ind))
        support_cand = torch.unique(support_cand)

        Aop.set_mask(support_cand)
        A_small = Aop.forward
        AT_small = Aop.transpose

        lsr_rec = least_squares(
            A_small,
            AT_small,
            b,
            z=0,
            init=x[support_cand].view(-1, 1),
            solver="minres",
            # gamma=1e5,
            tol=1e-5,
            max_iter=500,
            parallel_dim=-1,
            verbose=True,
        )

        lsr_flat = lsr_rec.flatten()
        _, max_ind = torch.topk(torch.abs(lsr_flat), k=sparsity)
        vals = lsr_flat[max_ind]
        support = support_cand[max_ind]
        res = b - A(vals.view(-1, 1), mask_indices=support)

        x.zero_()
        x[support] = vals

        res_norm = torch.linalg.vector_norm(res)
        if torch.abs(res_norm - res_norm_old) / torch.abs(res_norm + 1e-8) < tol:
            break
        if j % 5 == 0:
            print("Iteration:", j + 1, " Residual:", res_norm.item())
        res_norm_old = res_norm
    return x.view(-1, 1)


def aTchebychev_eval(D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,1), real valued
    #   p, k : tensors of type torch.Tensor and shapes (N,D)
    k_i = Vi(2, D)  # (M, 1, D) LazyTensor
    pre_i = Vi(3, 1)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(1, D)  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def Tchebychev_eval(D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,2), real-valued
    #   p, f : tensors of type torch.Tensor and shapes (N,D)
    k_j = Vj(2, D)  # (M, 1, D) LazyTensor
    pre_j = Vj(3, 1)  # (M, 1, 1) LazyTensor
    p_acos_i = Vi(1, D)  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)

    tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()
    return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def normalization_Techebychev(p_acos, k, pre, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,1), real valued
    #   p, k : tensors of type torch.Tensor and shapes (N,D)
    k_i = Vi(k)  # (M, 1, D) LazyTensor
    pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(p_acos)  # (1, N, D) LazyTensor

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return ((pre_i * tmp) ** 2).sum_reduction(dim=1, use_double_acc=True)


class MaskedOperator:
    def __init__(self, A, AT):
        self.A = A
        self.AT = AT
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return self.A(x, mask_indices=self.mask)

    def transpose(self, y):
        return self.AT(y, mask_indices=self.mask)
