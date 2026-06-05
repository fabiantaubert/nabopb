import numpy as np
import time
from multiprocessing import Pool
import NABOPB_subroutines as nabopb


def NABOPB_for_PDE(
    fct_handle,
    pde_dims,
    d,
    basis_flag,
    gamma=None,
    diminc=None,
    sparsity_s=None,
    sparsity_s_local=None,
    delta=1e-12,
    niter_r=2,
    pde_disc=3,
):
    """
    Dimension-incremental algorithm for nonlinear approximation of high-dimensional
    functions in a bounded orthonormal product basis.

    This method adaptively constructs an index set of a suitable basis support, ensuring
    that the largest basis coefficients are included. The algorithm is based on point
    evaluations of the function and supports various basis types and reconstruction methods.

    This is a modified version, only suitable for the approximation of solutions of
    differential equations.

    Parameters:
    -----------
    fct_handle : callable
        A function handle that accepts sampling nodes as an array of size [num_nodes, d].

    pde_dims : int
        Specifies amount of non-parameter dimensions (e.g. x and t) of the PDE.

    d : int
        Dimensionality of the function's domain.

    basis_flag : str
        Specifies the type of basis and reconstruction method. Options include:
        - 'Fourier_SRLasso+': Fourier basis on [0,1]^d with with square root LASSO with no/less oversampling.
        - 'Fourier_OMP+': Fourier basis on [0,1]^d with orthogonal matching pursuit with no/less oversampling.
        - 'Fourier_COSAMP+': Fourier basis on [0,1]^d with COSAMP with no/less oversampling.
        - 'Cheby_SRLasso+': Chebyshev basis on [-1,1]^d with with square root LASSO with no/less oversampling.
        - 'Cheby_OMP+': Chebyshev basis on [-1,1]^d with orthogonal matching pursuit with no/less oversampling.
        - 'Cheby_COSAMP+': Chebyshev basis on [-1,1]^d with COSAMP with no/less oversampling.

    gamma : dict, optional
        Specifies the search space for indices. Possible keys include:
        - 'N': Parameter defining the search space size (scalar or array of size d).
        - 'w': Weights defining the shape of the search space (scalar or array of size d).
        - 'superpos': Superposition dimension d_s (||k||_0 <= d_s).
        - 'sgn': Specifies index sign constraints ('default' or 'non-negative').
        - 'type': Specifies the grid type:
          - 'full': Full grid up to extension N in d dimensions.
          - 'sym_hc': Symmetric hyperbolic cross with extension N (scalar) and weights w.

    diminc : dict, optional
        Specifies the dimension-incremental strategy. Possible keys include:
        - 'workers': Maximum number of parallel workers.
        - 'type': Strategy type ('default' or 'dyadic').

    sparsity_s : int, optional
        Target sparsity of the output. Default is (2 * N_gamma + 1)^d.

    sparsity_s_local : int, optional
        Local target sparsity for intermediate steps. Defaults to `sparsity_s`.

    delta : float, optional
        Relative threshold for truncation at each step. Default is 1e-12.

    niter_r : int, optional
        Number of detection iterations. Default is 2.

    pde_disc : int, optional
        Number of virtual points per available pde_dim. Default is 3.

    Returns:
    --------
    index : ndarray
        Detected indices corresponding to the largest basis coefficients,
        of shape [#index, d].

    val : ndarray
        Corresponding basis coefficients.

    sample_sizes : list
        Number of samples used in each dimension-incremental step.

    cand_sizes : list
        Cardinality of candidate sets used.

    run_times : dict
        Computation times for various steps:
        - 'quadrature': Total time for construction of the sampling set.
        - 'sampling': Total time for sampling.
        - 'evaluation': Total time for computation of the coefficients.
        - 'total': Total runtime of the algorithm.

    References:
    -----------
    L. Kaemmerer, D. Potts, and F. Taubert, "Nonlinear approximation in bounded orthonormal product bases",
    Sampling Theory, Signal Processing, and Data Analysis, 21:19 (2023).
    DOI: https://doi.org/10.1007/s43670-023-00057-7

    S. Neumayer, D. Potts, and F. Taubert, "Learning solution operators of PDEs with sparse approximation",
    arXiv:2606.06046 (2026).
    DOI: https://arxiv.org/abs/2606.06046
    """

    # Default parameters
    if gamma is None:
        gamma = {"type": "full", "N": 32, "sgn": "default"}
    if diminc is None:
        diminc = {"type": "default"}
    if sparsity_s is None:
        sparsity_s = (2 * gamma["N"] + 1) ** d
    if sparsity_s_local is None:
        sparsity_s_local = sparsity_s

    # Parameter checks
    check_naturalnumber_or_error(d, "dimensionality d")
    check_naturalnumber_or_error(sparsity_s, "parameter s")
    check_naturalnumber_or_error(sparsity_s_local, "parameter s_local")
    check_0leq_param_or_error(delta, "parameter delta")
    check_naturalnumber_or_error(niter_r, "parameter niter_r")
    if "superpos" in gamma:
        check_naturalnumber_or_error(gamma["superpos"], "superposition dimension d_s")

    print(
        f'Algorithm Start: d = {d}, basis and method = {basis_flag}, diminc. approach = {diminc["type"]}'
    )
    print(f'Algorithm Start: Gamma type = {gamma["type"]}')
    print(
        f"Algorithm Start: sparsity s = {sparsity_s}, sparsity s_local = {sparsity_s_local}, delta = {delta:.3e}, det. iterations r = {niter_r}"
    )

    sample_sizes = np.zeros(d)
    cand_sizes = np.zeros(d)
    run_times = np.zeros(4)  # Construct, Sampling, Evaluation, All

    tAll = time.time()

    if diminc["type"] == "default":
        # Preallocation
        I = {(2, 2): []}
        # First 1D-detection
        print("Step 1:")
        I[(1, 1)], sample_sizes_j, cand_sizes_j, run_times_j = detect_1D(
            np.array([1]),
            fct_handle,
            pde_dims,
            d,
            gamma,
            basis_flag,
            sparsity_s_local,
            delta,
            niter_r,
            pde_disc,
        )
        sample_sizes[0] += sample_sizes_j
        cand_sizes[0] += cand_sizes_j
        run_times[:3] += run_times_j
        I[(1, 2)] = np.array([1])
        # Step 2
        print("Step 2:")
        for t in range(2, d + 1):
            # New 1D-detection
            I[(2, 1)], sample_sizes_j, cand_sizes_j, run_times_j = detect_1D(
                np.array([t]),
                fct_handle,
                pde_dims,
                d,
                gamma,
                basis_flag,
                sparsity_s_local,
                delta,
                niter_r,
                pde_disc,
            )
            sample_sizes[0] += sample_sizes_j
            cand_sizes[0] += cand_sizes_j
            run_times[:3] += run_times_j
            I[(2, 2)] = np.array([t])
            # Increment
            (
                I[(1, 1)],
                I[(1, 2)],
                val,
                sample_sizes_j,
                cand_sizes_j,
                run_times_j,
            ) = increment(
                I,
                fct_handle,
                pde_dims,
                d,
                gamma,
                basis_flag,
                sparsity_s,
                sparsity_s_local,
                delta,
                niter_r,
                pde_disc,
            )
            sample_sizes[t - 1] += sample_sizes_j
            cand_sizes[t - 1] += cand_sizes_j
            run_times[:3] += run_times_j
        sortIdx = np.argsort(I[(1, 2)])
        index = I[(1, 1)][:, sortIdx]
        print(f"Algorithm finished with {index.shape[0]} detected indices.\n\n")
        run_times[3] = time.time() - tAll
    # TO CHECK
    elif diminc["type"] == "dyadic":
        # Preallocation
        I = {(d, 2): []}
        # 1D-detections
        print("Step 1:")
        with Pool(diminc["workers"]) as pool:
            async_detections = [
                pool.apply_async(
                    detect_1D,
                    (
                        j,
                        fct_handle,
                        d,
                        gamma,
                        basis_flag,
                        sparsity_s_local,
                        delta,
                        niter_r,
                    ),
                )
                for j in range(1, d + 1)
            ]
            for j, detection in enumerate(async_detections):
                temp, sample_sizes_j, cand_sizes_j, run_times_j = detection.get()
                # I[(j + 1, :)] = temp, j + 1
                sample_sizes[0] += sample_sizes_j
                cand_sizes[0] += cand_sizes_j
                run_times[:3] += run_times_j
        # Step 2
        sample_sizes_j = 0
        cand_sizes_j = 0
        run_times_j = np.zeros(3)
        J = {(1, 2): []}
        while len(I) > 2:
            with Pool(diminc["workers"]) as pool:
                async_increments = [
                    pool.apply_async(
                        increment,
                        (
                            I[2 * k - 1],
                            I[2 * k],
                            fct_handle,
                            d,
                            gamma,
                            basis_flag,
                            sparsity_s,
                            sparsity_s_local,
                            delta,
                            niter_r,
                        ),
                    )
                    for k in range(1, len(I) // 2 + 1)
                ]
                for k, increment_result in enumerate(async_increments):
                    (
                        tempA,
                        tempB,
                        _,
                        sample_sizes_j,
                        cand_sizes_j,
                        run_times_j,
                    ) = increment_result.get()
                    # J[(k + 1, :)] = tempA, tempB
                    sample_sizes[len(J[(k + 1, 2)])] += sample_sizes_j
                    cand_sizes[len(J[(k + 1, 2)])] += cand_sizes_j
                    run_times[:3] += run_times_j
            if len(I) % 2 == 1:
                # J[(len(J) + 1, :)] = I[len(I) - 1]
                pass
            len_J = [len(J[(l + 1, 2)]) for l in range(len(J))]
            sortIdx = np.argsort(len_J)
            I = {(l + 1, 1): J[(sortIdx[l] + 1, 1)] for l in range(len(sortIdx))}
            J = {(1, 2): []}
        # Final increment
        tempA, tempB, val, sample_sizes_j, cand_sizes_j, run_times_j = increment(
            I,
            fct_handle,
            d,
            gamma,
            basis_flag,
            sparsity_s,
            sparsity_s_local,
            delta,
            niter_r,
        )
        # J[(1, :)] = tempA, tempB
        sample_sizes[d - 1] += sample_sizes_j
        cand_sizes[d - 1] += cand_sizes_j
        run_times[:3] += run_times_j
        sortIdx = np.argsort(J[(1, 2)])
        index = J[(1, 1)][:, sortIdx]
        print(f"Algorithm finished with {index.shape[0]} detected indices.\n\n")
        run_times[3] = time.time() - tAll
    else:
        raise ValueError(
            f'{diminc["type"]} as dimension-incremental approach is not defined.'
        )

    result = {
        "index": index,
        "val": val,
        "sample_sizes": sample_sizes,
        "cand_sizes": cand_sizes,
        "run_times": run_times,
    }
    return result


def check_naturalnumber_or_error(param, name):
    if param < 1 or not isinstance(param, int):
        raise ValueError(f"{name} must be a natural number")


def check_0leq_param_or_error(param, name):
    if param < 0:
        raise ValueError(f"{name} must be >= 0")


def increment(
    J,
    fct_handle,
    pde_dims,
    d,
    gamma,
    basis_flag,
    sparsity_s,
    sparsity_s_local,
    delta,
    niter_r,
    pde_disc,
):
    sample_sizes = 0
    cand_sizes = 0
    run_times = np.zeros(3)
    val2 = 0
    # Construction of the candidate set K
    K = []
    ma = J[(1, 1)].shape[0]
    mb = J[(2, 1)].shape[0]
    a, b = np.meshgrid(np.arange(ma), np.arange(mb))
    product = np.hstack((J[(1, 1)][a.flatten(), :], J[(2, 1)][b.flatten(), :]))
    dims = np.concatenate((J[(1, 2)], J[(2, 2)]))  # Notation of the dimensions
    if gamma["type"] == "full":
        K = product
    elif gamma["type"] == "sym_hc":
        if isinstance(gamma["w"], int):
            temp = np.abs(product) / gamma["w"]
        else:
            weights = gamma["w"][dims]
            temp = np.outer(np.ones(product.shape[0]), weights) * np.abs(product)
        temp[temp <= 1] = 1
        product_idx = 2 ** gamma["N"] >= np.prod(temp, axis=1)
        K = product[product_idx, :]
    else:
        raise ValueError(f'{gamma["type"]} as Gamma type is not defined.')
    # Check for superposition
    if "superpos" in gamma:
        superpos_idx = np.sum(K != 0, axis=1) <= gamma["superpos"]
        K = K[superpos_idx, :]
        if K.size == 0:
            raise ValueError(
                "Superposition assumption resulted in an empty candidate set K!"
            )
    cand_sizes += K.shape[0]
    # Construction of the sampling set and the cubature
    tConstruct = time.time()
    W, Xi = nabopb.construct_cubature(
        K, basis_flag, dims, d, sparsity_s_local, pde_dims
    )
    M = Xi.shape[0]
    run_times[0] += time.time() - tConstruct
    I = []
    if len(dims) < d:
        for r in range(niter_r):
            x_tilde = nabopb.random_number(d, dims, basis_flag)
            x_tilde = np.outer(np.ones(Xi.shape[0]), x_tilde)
            x = np.hstack((Xi, x_tilde))
            _, sortIdx = np.unique(
                np.hstack((dims - 1, np.delete(np.arange(d), dims - 1))),
                return_index=True,
                axis=0,
            )
            x = x[:, sortIdx]
            # Sampling
            tSampling = time.time()
            if max(dims) > pde_dims:
                sample_dims = pde_dims - np.sum(dims <= pde_dims)
                if sample_dims == 0:
                    L = pde_disc ** (pde_dims)
                    Xi_temp = np.repeat(Xi, L + 1, axis=0)
                    for i in range(M):
                        eval_grid = nabopb.random_number(
                            pde_dims,
                            np.empty(0),
                            basis_flag,
                            number_points=pde_disc ** (pde_dims),
                        )
                        Xi_temp[i * (L + 1) : i * (L + 1) + L, :pde_dims] = eval_grid
                    f = fct_handle(
                        Xi_temp[:, :pde_dims], x[:, pde_dims:], blocksize=L + 1
                    )
                    sample_sizes += x.shape[0]
                else:
                    # TO BE IMPLEMENTED
                    raise ValueError("Dyadic and others not yet implemented.")
            else:
                f = fct_handle(x[:, :pde_dims], x[0:1, pde_dims:], blocksize=M)
                Xi_temp = Xi
                sample_sizes += 1
            run_times[1] += time.time() - tSampling
            print(Xi.shape)
            # Computation of the projected coefficients
            tEvaluate = time.time()
            f_hat = nabopb.evaluate_cubature(W, Xi_temp, f, K, basis_flag, dims, d)
            run_times[2] += time.time() - tEvaluate
            # Sorting and Thresholding
            sortIdx = np.argsort(np.abs(f_hat))[::-1]
            sortIdx = sortIdx[np.abs(f_hat[sortIdx]) >= delta]
            if len(sortIdx) > sparsity_s_local:
                sortIdx = sortIdx[:sparsity_s_local]
            # Adding detected indices to I
            if len(I) == 0:
                I = K[sortIdx, :]
            else:
                I = np.unique(np.vstack((I, K[sortIdx, :])), axis=0)
        print(f"dim-inc of [{J[(1, 2)]}] and [{J[(2, 2)]}] done, card(I)={I.shape[0]}")
    else:
        sortIdx = np.argsort(dims)
        x = Xi[:, sortIdx]
        tSampling = time.time()
        L = pde_disc ** (pde_dims)
        Xi_temp = np.repeat(Xi, L + 1, axis=0)
        for i in range(M):
            eval_grid = nabopb.random_number(
                pde_dims, np.empty(0), basis_flag, number_points=pde_disc ** (pde_dims)
            )
            Xi_temp[i * (L + 1) : i * (L + 1) + L, :pde_dims] = eval_grid
        f = fct_handle(Xi_temp[:, :pde_dims], x[:, pde_dims:], blocksize=L + 1)
        run_times[1] += time.time() - tSampling
        tEvaluate = time.time()
        f_hat = nabopb.evaluate_cubature(W, Xi_temp, f, K, basis_flag, dims, d)
        run_times[2] += time.time() - tEvaluate
        sortIdx = np.argsort(np.abs(f_hat))[::-1]
        sortIdx = sortIdx[np.abs(f_hat[sortIdx]) >= delta]
        if len(sortIdx) > sparsity_s:
            sortIdx = sortIdx[:sparsity_s]
        I = K[sortIdx, :]
        val2 = f_hat[sortIdx]
        sample_sizes += x.shape[0]
    return I, dims, val2, sample_sizes, cand_sizes, run_times


def detect_1D(
    j,
    fct_handle,
    pde_dims,
    d,
    gamma,
    basis_flag,
    sparsity_s_local,
    delta,
    niter_r,
    pde_disc,
):
    sample_sizes = 0
    cand_sizes = 0
    run_times = np.zeros(3)
    # Construction of the candidate set K
    K = 0
    if gamma["type"] == "full":
        if isinstance(gamma["N"], int):
            ext = gamma["N"]
        else:
            ext = gamma["N"][j - 1]
        if gamma["sgn"] == "default":
            K = np.arange(-ext, ext + 1).reshape(-1, 1)
        elif gamma["sgn"] == "non-negative":
            K = np.arange(0, ext + 1).reshape(-1, 1)
        else:
            raise ValueError(f'{gamma["sgn"]} as Gamma Signum is not defined.')
    elif gamma["type"] == "sym_hc":
        if isinstance(gamma["w"], int):
            weight = gamma["w"]
        else:
            weight = gamma["w"][j - 1]
        max_val = 2 ** gamma["N"] * weight
        if gamma["sgn"] == "default":
            K = np.arange(-max_val, max_val + 1).reshape(-1, 1)
        elif gamma["sgn"] == "non-negative":
            K = np.arange(0, max_val + 1).reshape(-1, 1)
        else:
            raise ValueError(f'{gamma["sgn"]} as Gamma Signum is not defined.')
    else:
        raise ValueError(f'{gamma["type"]} as Gamma type is not defined.')
    cand_sizes += K.shape[0]
    # Construction of the sampling set and the cubature
    tConstruct = time.time()
    W, Xi = nabopb.construct_cubature_1d(K, basis_flag, j, d, sparsity_s_local)
    Xi = np.expand_dims(Xi, axis=1)
    M = Xi.shape[0]
    run_times[0] += time.time() - tConstruct
    for r in range(niter_r):
        x_tilde = nabopb.random_number(d, j, basis_flag)
        x_tilde = np.outer(np.ones(Xi.shape[0]), x_tilde)
        if j == 1:
            x = np.hstack((Xi, x_tilde))
        elif j == d:
            x = np.hstack((x_tilde, Xi))
        else:
            x = np.hstack((x_tilde[:, : int(j - 1)], Xi, x_tilde[:, int(j - 1) :]))
        # Sampling
        if j > pde_dims:
            L = 1
            eval_grid = x[:, :pde_dims]
        else:
            eval_grid = nabopb.random_number(
                pde_dims - 1,
                np.empty(0),
                basis_flag,
                number_points=pde_disc ** (pde_dims - 1),
            )
            L, _ = eval_grid.shape
            E = np.repeat(eval_grid, M, axis=0)
            X = np.tile(Xi, (L, 1))
            eval_grid = np.insert(E, j - 1, X, axis=1)
        tSampling = time.time()
        if j > pde_dims:
            f = fct_handle(eval_grid, x[:, pde_dims:], blocksize=1)
            sample_sizes += x.shape[0]
        else:
            f = fct_handle(eval_grid, x[0:1, pde_dims:], blocksize=M * L)
            sample_sizes += 1
        run_times[1] += time.time() - tSampling
        # Computation of the projected coefficients
        for i in range(L):
            tEvaluate = time.time()
            if j > pde_dims:
                f_hat = nabopb.evaluate_cubature_1d(W, Xi, f, K, basis_flag, j, d)
            else:
                f_hat = nabopb.evaluate_cubature_1d(
                    W, Xi, f[i * M : (i + 1) * M], K, basis_flag, j, d
                )
            run_times[2] += time.time() - tEvaluate
            # Sorting and Thresholding
            sortIdx = np.argsort(np.abs(f_hat))[::-1]
            sortIdx = sortIdx[np.abs(f_hat[sortIdx]) >= delta]
            if len(sortIdx) > sparsity_s_local:
                sortIdx = sortIdx[:sparsity_s_local]
            # Adding detected indices to I
            if r == 0 and i == 0:
                I = K[sortIdx]
            else:
                I = np.vstack((I, K[sortIdx]))
                I = np.unique(I, axis=0)
    print(f"dim {j} done, card(I)={I.shape[0]}")
    return I, sample_sizes, cand_sizes, run_times
