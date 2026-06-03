import math
import torch
from pykeops.torch import Vi, Vj
from deepinv.optim.utils import least_squares


def COSAMP_Fourier(
    samples, values, frequencies, sparsity=5000, num_iters=100, tol=1e-7
):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points

    normalization = torch.sqrt(
        normalization_fourier(samples, frequencies) / N
    )  # (M, 2)

    A_handle = aNUDFT(samples, D)
    AT_handle = NUDFT(samples, D)

    def A(x, mask_indices=None):
        if mask_indices is None:
            f = frequencies
        else:
            f = frequencies[mask_indices]
        return A_handle(x, f) / math.sqrt(N)

    def AT(x, mask_indices=None):
        if mask_indices is None:
            f = frequencies
        else:
            f = frequencies[mask_indices]
        return AT_handle(x, f) / math.sqrt(N)

    b = values / math.sqrt(N)

    norm = torch.linalg.norm(b)
    # run COSAMP reconstruction
    rec = COSAMP_Fourier_grouped(
        A,
        AT,
        normalization,
        b,
        frequencies,
        samples,
        sparsity=sparsity,
        num_iters=num_iters,
        tol=tol * norm,
    )
    residuals = torch.linalg.norm(A(rec) - b)
    print("Final residual:", residuals.item())
    rec_complex = rec[:, 0] + 1j * rec[:, 1]
    rec_complex = rec_complex.unsqueeze(1)
    return rec_complex, residuals


def COSAMP_Fourier_grouped(
    A, AT, normalization, b, f, p, sparsity=5000, num_iters=100, tol=1e-3
):
    device = p.device
    dtype = p.dtype

    M = f.size(0)
    sparsity = min(sparsity, M)
    two_sparsity = min(2 * sparsity, M)

    eps = 1e-10

    x = torch.zeros((M, 2), device=device, dtype=dtype)
    support = torch.empty(0, device=device, dtype=torch.long)

    norm_freq = normalization.norm(dim=-1).clamp(min=eps)
    inv_norm_freq = 1.0 / norm_freq

    res = b.clone()
    res_norm_old = torch.tensor(float("inf"), device=device, dtype=dtype)
    Aop = MaskedOperator(A, AT)

    for j in range(num_iters):
        proxy = AT(res)  # (M, 2)
        score = proxy.norm(dim=-1) * inv_norm_freq  # (M,)

        _, ind = torch.topk(score, k=two_sparsity)

        support_cand = torch.cat((support, ind))
        support_cand = torch.unique(support_cand)

        Aop.set_mask(support_cand)
        A_small = Aop.forward
        AT_small = Aop.transpose

        init = x[support_cand, :]  # (2|T|, 1)

        lsr_rec = least_squares(
            A_small,
            AT_small,
            b,
            z=0,
            init=init,
            solver="minres",
            tol=1e-5,
            max_iter=500,
            parallel_dim=None,
            verbose=True,
        )

        cand_score = lsr_rec.norm(dim=-1)
        keep = min(sparsity, cand_score.numel())
        _, max_ind = torch.topk(cand_score, k=keep)

        support = support_cand[max_ind]
        vals = lsr_rec[max_ind, :]

        x.zero_()
        x[support, :] = vals

        res = b - A(vals, mask_indices=support)

        res_norm = torch.linalg.vector_norm(res)
        if torch.abs(res_norm - res_norm_old) / (torch.abs(res_norm) + 1e-8) < tol:
            break

        if j % 5 == 0:
            print("Iteration:", j + 1, " Residual:", res_norm.item())

        res_norm_old = res_norm

    return x


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


def NUDFT(p, D):
    f_i = Vi(1, D)
    p_j = Vj(p)
    x_j = Vj(0, 1)
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) * x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def aNUDFT(p, D):
    f_j = Vj(1, D)
    p_i = Vi(p)
    x_j = Vj(0, 2)
    return ((f_j | p_i).unary("ComplexExp1j", dimres=2) | x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def normalization_fourier(p, f):
    f_i = Vi(f)
    p_j = Vj(p)
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) ** 2).sum_reduction(
        dim=1, use_double_acc=True
    )
