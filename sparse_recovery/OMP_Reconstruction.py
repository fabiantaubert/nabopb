import math
import torch
import numpy as np
from pykeops.torch import Vi, Vj


def build_explicit_A(samples, frequencies, dtype=None, eps=1e-12):
    if dtype is None:
        dtype = samples.dtype

    samples = samples.to(torch.double)
    frequencies = frequencies.to(torch.double)

    phase = samples @ frequencies.T  # (N, M)

    A_cos = torch.cos(phase)
    A_sin = torch.sin(phase)

    cols = []
    atom_map = []

    M = frequencies.shape[0]
    for m in range(M):
        cos_col = A_cos[:, m : m + 1]
        cols.append(cos_col)
        atom_map.append((m, 0))  # 0 = cos

        sin_col = A_sin[:, m : m + 1]
        if torch.linalg.norm(sin_col) > eps:
            cols.append(sin_col)
            atom_map.append((m, 1))  # 1 = sin

    A = torch.cat(cols, dim=1).to(dtype)
    return A, atom_map


def OMP_Fourier(samples, values, frequencies, num_iters=2000, tol=5e-6):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    A_handle = aNUDFT(samples, D)
    AT_handle = NUDFT(samples, D)

    def A(x, mask_indices=None):
        if mask_indices is None:
            f = frequencies
            return A_handle(x, f)
        else:
            # only evaluate selected frequencies
            ind_freq = mask_indices >> 1
            f = frequencies[ind_freq, :]
            # handle real and imaginary parts
            ind_y = mask_indices & 1
            tmp = torch.zeros((x.size(0), 2), dtype=dtype, device=device)
            tmp.scatter_(1, ind_y.unsqueeze(1), x)
            return A_handle(tmp, f)

    def AT(x, mask_indices=None):
        if mask_indices is None:
            f = frequencies
            return AT_handle(x, f)
        else:
            # only evaluate selected frequencies
            ind_freq = mask_indices >> 1
            f = frequencies[ind_freq, :]
            result = AT_handle(x, f)
            # extract correct real or imaginary part
            ind_y = mask_indices & 1
            return torch.gather(result, 1, ind_y.unsqueeze(1))

    def col_extractor(samples, frequencies, max_ind):
        # build complex matrix for least squares problem explicitly
        ind_x, ind_y = max_ind >> 1, max_ind & 1
        mat = (samples.to(torch.double) * frequencies[ind_x, :].to(torch.double)).sum(
            -1
        )
        if ind_y == 0:
            return torch.cos(mat)[:, None].to(dtype)
        else:
            return torch.sin(mat)[:, None].to(dtype)

    b = values
    norm = torch.linalg.norm(b)
    normalization = torch.sqrt(normalization_fourier(samples, frequencies))

    # run OMP reconstruction — selects full (cos, sin) frequency pairs jointly
    rec, nnz = OMP2_Fourier(
        col_extractor,
        A,
        AT,
        normalization,
        b,
        frequencies,
        samples,
        num_iters=num_iters,
        tol=tol * norm,
    )

    rec_complex = rec[:, 0] + 1j * rec[:, 1]
    rec_complex = rec_complex.unsqueeze(1)

    residuals = torch.linalg.norm(A(rec) - b)
    print("Final residual:", residuals.item(), "Nonzero amount:", nnz)
    return rec_complex, residuals


def OMP2_Fourier(
    col_extractor, A, AT, normalization, b, f, p, num_iters=1500, tol=1e-7
):
    """
    OMP for Fourier dictionaries that selects full (cos, sin) frequency pairs
    jointly. At each iteration the frequency k maximising
        ||(corr_cos[k], corr_sin[k])|| / norm_freq[k]
    is chosen, after which the Cholesky factor is extended with two sequential
    rank-1 updates — one for the cos atom and one for the sin atom — reusing
    all existing machinery unchanged.

    Args:
        col_extractor : callable(p, f, atom_idx) -> (N, 1) column
        A             : forward operator, A(x, mask_indices=None)
        AT            : adjoint operator, AT(x, mask_indices=None)
        normalization : (M, 2) tensor — sqrt column norms, [:, 0] cos, [:, 1] sin
        b             : (N, 1) measurement vector
        f             : (M, D) frequency array
        p             : (N, D) sample points
        num_iters     : maximum number of *frequency* iterations
        tol           : residual stopping threshold (absolute)
    """
    device = p.device
    dtype = p.dtype
    # eps = torch.finfo(dtype).eps
    eps = 1e-9
    N = b.size(0)
    M = normalization.size(0)

    A_mat, atom_map = build_explicit_A(p, f, dtype=torch.double)

    print("A shape:", A_mat.shape)
    print("rank:", torch.linalg.matrix_rank(A_mat).item())
    print("cond(A):", torch.linalg.cond(A_mat).item())
    # print("cond(A^T A):", torch.linalg.cond(A_mat.T @ A_mat).item())
    # print("cond(A A^T):", torch.linalg.cond(A_mat @ A_mat.T).item())

    # Per-frequency Frobenius norm of the [col_cos | col_sin] block, shape (M,)
    norm_freq = normalization.norm(dim=-1).clamp(min=eps)

    # Per-atom norms, flattened to (2M,), used for the Cholesky diagonal entries
    diag = normalization.flatten().clamp(min=eps)

    # Solution tensor: x[k, 0] = cos coeff, x[k, 1] = sin coeff
    x = torch.zeros((M, 2), dtype=dtype, device=device)

    # Cap iterations: each frequency contributes 2 atoms, stay within budget
    max_freq_iters = min(num_iters, M, math.floor(b.size(0) / 2))
    print(max_freq_iters)
    max_atoms = 2 * max_freq_iters

    selected_freq = torch.zeros(max_freq_iters, device=device, dtype=torch.long)
    selected_atoms = torch.zeros(max_atoms, device=device, dtype=torch.long)

    L = torch.zeros((max_atoms, max_atoms), device=device, dtype=dtype)
    rhs = torch.zeros((max_atoms, 1), device=device, dtype=dtype)

    res = b
    out = torch.zeros((0, 1), device=device, dtype=dtype)
    actual_atoms = 0
    actual_iters = 0
    n = 0

    for j in range(max_freq_iters):
        # ------------------------------------------------------------------
        # 1. Select frequency with largest normalised complex correlation
        # ------------------------------------------------------------------
        corr_full = AT(res)  # (M, 2)
        score = corr_full.norm(dim=-1) / norm_freq  # (M,)
        if j > 0:
            score[selected_freq[:j]] = -1.0  # mask already chosen freqs
        max_freq = torch.argmax(score)
        selected_freq[j] = max_freq

        atom_cos = 2 * max_freq  # index of the cos atom
        atom_sin = 2 * max_freq + 1  # index of the sin atom
        # n = 2 * j                        # number of atoms already in L

        # ------------------------------------------------------------------
        # 2a. Rank-1 Cholesky update for the cos atom
        # ------------------------------------------------------------------
        col_cos = col_extractor(p, f, atom_cos)  # (N, 1)
        selected_atoms[n] = atom_cos

        if n == 0:
            L[0, 0] = col_cos.norm()
        else:
            w = AT(col_cos, mask_indices=selected_atoms[:n])  # (n, 1)
            v = torch.linalg.solve_triangular(L[:n, :n], w, upper=False)
            L[n, :n] = v.T
            L[n, n] = torch.sqrt(
                torch.clamp(diag[atom_cos] ** 2 - v.pow(2).sum(), min=eps**2)
            )

        rhs[n] = torch.dot(col_cos.flatten(), b.flatten())
        n += 1

        # ------------------------------------------------------------------
        # 2b. Rank-1 Cholesky update for the sin atom
        #     Note: selected_atoms[:n+1] now includes atom_cos above
        # ------------------------------------------------------------------
        col_sin = col_extractor(p, f, atom_sin)  # (N, 1)

        if torch.norm(col_sin) > eps:
            selected_atoms[n] = atom_sin

            w = AT(col_sin, mask_indices=selected_atoms[:n])  # (n+1, 1)
            v = torch.linalg.solve_triangular(L[:n, :n], w, upper=False)
            L[n, :n] = v.T
            L[n, n] = torch.sqrt(
                torch.clamp(diag[atom_sin] ** 2 - v.pow(2).sum(), min=eps**2)
            )

            rhs[n] = torch.dot(col_sin.flatten(), b.flatten())
            n += 1

        # ------------------------------------------------------------------
        # 3. Solve updated normal equations and recompute residual
        # ------------------------------------------------------------------
        actual_atoms = n
        out = torch.cholesky_solve(rhs[:actual_atoms], L[:actual_atoms, :actual_atoms])
        res = b - A(out, mask_indices=selected_atoms[:actual_atoms])

        residual = res.norm() / math.sqrt(N)
        actual_iters = j + 1
        if j % 100 == 0:
            print("Iteration:", actual_iters, " Residual:", residual.item())
        if residual < tol:
            break

    x.flatten()[selected_atoms[:actual_atoms]] = out[:actual_atoms].flatten()
    return x, actual_iters


def OMP_Chebyshev(samples, values, indices, num_iters=2000, tol=5e-6):
    D = samples.size(1)  # dimension of sample points
    dtype = samples.dtype

    # precompute normalization coefficients for computational efficiency
    norm_coeffs = math.sqrt(2) ** (indices.clamp(0, 1).sum(-1, keepdim=True))

    # store samples in acos format
    samples_acos = torch.acos(samples)

    normalization = torch.sqrt(
        normalization_Techebychev(samples_acos, indices, norm_coeffs, D)
    )

    A_handle = Tchebychev_eval(samples_acos, D)
    AT_handle = aTchebychev_eval(samples_acos, D)

    def A(x, mask_indices=None):
        if mask_indices == None:
            f = indices
            pre = norm_coeffs
        else:
            f = indices[mask_indices]
            pre = norm_coeffs[mask_indices]
        return A_handle(x, f, pre)

    def AT(x, mask_indices=None):
        if mask_indices == None:
            f = indices
            pre = norm_coeffs
        else:
            f = indices[mask_indices]
            pre = norm_coeffs[mask_indices]
        return AT_handle(x, f, pre)

    def col_extractor(samples_acos, indices, max_ind):
        # build complex matrix for least squares problem explicitly
        mat = norm_coeffs[max_ind, 0] * torch.cos(
            indices[max_ind, :].to(torch.double) * samples_acos.to(torch.double)
        ).prod(-1)
        return mat[:, None].to(dtype)

    b = values
    norm = torch.linalg.norm(b)
    # run OMP reconstruction
    rec, nnz = OMP2(
        col_extractor,
        A,
        AT,
        normalization,
        b,
        indices,
        samples_acos,
        num_iters=num_iters,
        tol=tol * norm,
    )
    residuals = torch.linalg.norm(A(rec) - b) / math.sqrt(b.size(0))
    print("Final residual:", residuals.item(), "Nonzero amount:", nnz)
    return rec, residuals


def OMP2(col_extractor, A, AT, normalization, b, f, p, num_iters=1500, tol=1e-7):
    device = p.device
    dtype = p.dtype
    eps = torch.finfo(dtype).eps
    N = b.size(0)
    corr = AT(b)
    x = torch.zeros_like(corr, device=device, dtype=dtype, requires_grad=False)

    selected_indices = torch.zeros(
        num_iters, device=device, dtype=torch.long, requires_grad=False
    )

    L = torch.zeros((num_iters, num_iters), device=device, dtype=dtype)
    rhs = torch.zeros((num_iters, 1), device=device, dtype=dtype)
    diag = torch.clamp(normalization.flatten(), min=eps)
    z_2 = torch.empty_like(diag)
    res = b
    num_iters = min(num_iters, x.numel() - 1)
    num_iters = math.floor(min(num_iters, b.size(0)))

    for j in range(num_iters):
        # Compute and mask correlations
        torch.abs(AT(res).flatten() / diag, out=z_2)
        if j > 0:
            z_2[selected_indices[:j]] = -1

        # Find maximum correlation
        max_ind = torch.argmax(z_2)
        selected_indices[j] = max_ind

        # Recursive construction of Cholesky decomposition
        # See https://ieeexplore.ieee.org/document/6333943/
        new_col = col_extractor(p, f, max_ind)

        if j == 0:
            L[0, 0] = torch.linalg.vector_norm(new_col)
        else:
            corr = AT(new_col, mask_indices=selected_indices[:j])
            v = torch.linalg.solve_triangular(L[:j, :j], corr, upper=False)
            L[j, :j] = v.T
            L[j, j] = torch.sqrt(
                torch.clamp(diag[max_ind] ** 2 - torch.sum(v**2), min=eps**2)
            )

        rhs[j, :] = torch.dot(new_col.flatten(), b.flatten())
        out = torch.cholesky_solve(rhs[: j + 1, :], L[: j + 1, : j + 1])
        res = b - A(out[: j + 1], mask_indices=selected_indices[: j + 1])

        residual = torch.linalg.vector_norm(res) / math.sqrt(N)
        if j % 100 == 0:
            print("Iteration:", j + 1, " Residual:", residual.item())
        if residual < tol:
            num_iters = j + 1
            break

    # Update solution using selected indices
    x.flatten()[selected_indices[:num_iters]] = out[:num_iters].flatten()

    return x, num_iters


def OMP(col_extractor, AT, normalization, b, f, p, num_iters=1500, tol=1e-7):
    if b.dtype == torch.float:
        safeguard = 1e-5
    else:
        safeguard = 1e-11
    device = p.device
    dtype = p.dtype

    x = torch.zeros_like(AT(b), device=device, dtype=dtype, requires_grad=False)
    indices = torch.zeros_like(
        AT(b), device=device, dtype=bool, requires_grad=False
    ).flatten()
    out_ind = torch.zeros(num_iters, device=device, dtype=int, requires_grad=False)
    A_mat = torch.zeros(b.numel(), num_iters, device=device, dtype=dtype)
    L = torch.zeros((num_iters, num_iters), device=device, dtype=dtype)
    rhs = torch.zeros((num_iters, 1), device=device, dtype=dtype)
    diag = normalization.flatten()
    res = b

    num_iters = min(num_iters, x.size(0))
    for j in range(num_iters):
        if j % 50 == 0:
            print(
                "Iteration:", j + 1, " Residual:", torch.linalg.vector_norm(res).item()
            )
        if (torch.linalg.vector_norm(res)) < tol:
            num_iters = j
            break

        z_2 = AT(res).flatten() / (diag + safeguard)
        z_2[indices] = 0
        max_ind = torch.argmax(torch.abs(z_2))
        out_ind[j] = max_ind
        indices[max_ind] = True

        # Recursive construction of Cholesky decomposition
        # See https://ieeexplore.ieee.org/document/6333943/
        if j == 0:
            new_col = col_extractor(p, f, max_ind)
            A_mat[:, j] = new_col.squeeze()
            L[0, 0] = torch.sqrt(torch.sum(new_col**2))
        else:
            new_col = col_extractor(p, f, max_ind)
            v = torch.linalg.solve_triangular(
                L[:j, :j], A_mat[:, :j].transpose(0, 1) @ new_col, upper=False
            )
            L[j, :j] = v.T
            L[j, j] = torch.sqrt(
                torch.clamp(diag[max_ind] ** 2 - torch.sum(v**2), min=safeguard)
            )

            A_mat[:, j] = new_col.squeeze()
        rhs[j, :] = torch.sum(new_col * b)
        out = torch.cholesky_solve(rhs[: j + 1, :], L[: j + 1, : j + 1])
        res = b - A_mat[:, : j + 1] @ out
    x_flat = x.flatten()
    x_flat.index_copy_(0, out_ind[:num_iters], out[:num_iters].squeeze())
    x = x_flat.view_as(x)
    return x, num_iters


def NUDFT(p, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, f : tensors of type torch.Tensor and shapes (N,D), (M,D)
    f_i = Vi(1, D)  # (M, 1, D) LazyTensor
    p_j = Vj(p)  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)  # (1, N, 1) LazyTensor
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) * x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def aNUDFT(p, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,2), real-valued
    # p, f : tensors of type torch.Tensor and shapes (N,D), (M,D)
    f_j = Vj(1, D)  # (1, M, D) LazyTensor
    p_i = Vi(p)  # (N, 1, D) LazyTensor
    x_j = Vj(0, 2)  # (1, M, 2) LazyTensor
    return ((f_j | p_i).unary("ComplexExp1j", dimres=2) | x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def normalization_fourier(p, f):
    # normalization of matrix columns
    f_i = Vi(f)
    p_j = Vj(p)
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) ** 2).sum_reduction(
        dim=1, use_double_acc=True
    )


def aTchebychev_eval(p_acos, D):
    # Adjoint Techebychev Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
    k_i = Vi(1, D)  # (M, 1, D) LazyTensor
    pre_i = Vi(2, 1)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(p_acos)  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)  # (1, N, 1) LazyTensor

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def Tchebychev_eval(p_acos, D):
    # Techebychev Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
    k_j = Vj(1, D)  # (1, M, D) LazyTensor
    pre_j = Vj(2, 1)  # (1, M, 1) LazyTensor
    p_acos_i = Vi(p_acos)  # (N, 1, D) LazyTensor
    x_j = Vj(0, 1)  # (1, M, 1) LazyTensor

    tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()
    return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def normalization_Techebychev(p_acos, k, pre, D):
    # normalization of matrix columns
    k_i = Vi(k)  # (M, 1, D) LazyTensor
    pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(p_acos)  # (1, N, D) LazyTensor

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return ((pre_i * tmp) ** 2).sum_reduction(dim=1, use_double_acc=True)
