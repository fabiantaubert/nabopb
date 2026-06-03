import math
import torch
import numpy as np
from tqdm.notebook import tqdm


def pd_rLASSO(
    x0,
    y0,
    A,
    AT,
    b,
    tau=1.0,
    sigma=1.0,
    num_iters=250,
    lam=1.0,
    tol=1e-3,
):
    x, y = x0, y0
    vals, vals_dual = [], []
    SoftShrink = torch.nn.Softshrink(lambd=tau * lam)

    eps = torch.finfo(x0.dtype).eps

    # iterator = tqdm(range(num_iters), leave=False)
    for j in range(num_iters):
        # primal dual iterations
        x_old = torch.clone(x)  # store previous iterate
        x = SoftShrink(x - tau * AT(y))
        yshift = y + sigma * A(2 * x - x_old) - sigma * b
        norm_yshift = torch.linalg.norm(yshift, ord=2, dim=0, keepdim=True)
        y = yshift / torch.clamp(norm_yshift, min=1.0)

        # tracking primal objective
        vals.append(
            (torch.linalg.norm(A(x) - b) + lam * torch.sum(torch.abs(x))).item()
        )

        # lower bound dual objective by constructing feasible y
        dual_arg = y * torch.clamp(lam / torch.max(torch.abs(AT(y))), max=1.0)
        dual_arg *= torch.clamp(
            1.0 / (torch.linalg.norm(dual_arg, ord=2, dim=0)), max=1.0
        )
        vals_dual.append((-torch.sum(dual_arg * b)).item())

        # check convergence based on optimality gap
        if vals[-1] - vals_dual[-1] < max(tol, eps):
            break

    print(j)
    print(vals[-1] - vals_dual[-1])

    return x, y, vals, vals_dual


def restart_pd_rLASSO(x0, A, AT, b, lam=1.0, tol=1e-5, restarts=10, beta=2, alpha=5):
    res = torch.randn_like(x0)
    for _ in range(100):
        res = AT(A(res))
        res /= torch.linalg.norm(res)
    norm = torch.sqrt(torch.linalg.norm(AT(A(res))))
    # print("Estimated norm of A is", norm.item())
    tau, sigma = 0.99 / norm, 0.99 / norm

    r = np.exp(-1)
    alpha = alpha * norm  # sharpness parameters beta=2 wellposed or beta=3 illposed
    eps = torch.linalg.norm(A(x0) - b) + lam * torch.sum(torch.abs(x0))
    x, y = x0, torch.zeros_like(b, requires_grad=False)
    vals, vals_dual = [], []

    for t in range(restarts):
        delta = (2 * eps / alpha) ** (1 / beta)
        eps = r * eps
        if eps < tol:
            #            print("Machine precision reached.")
            break
        K = torch.ceil(2 * delta * norm / eps).int()
        # print("Starting PDHGM call", t + 1, " out of", restarts)
        x_new, y_new, vals_alg, vals_dual_alg = pd_rLASSO(
            x,
            y,
            A,
            AT,
            b,
            tau=tau * delta,
            sigma=sigma / delta,
            num_iters=K,
            lam=lam,
            tol=eps,
        )
        if not vals or (vals_alg[-1] < vals[-1]):
            vals += [i for i in vals_alg]
            vals_dual += [i for i in vals_dual_alg]
            x = x_new
            y = y_new
    return x, vals, vals_dual


def restart_ls_pd_rLASSO(
    x0,
    A,
    AT,
    b,
    lam=1.0,
    schedule_length=15000,
    device_id="cuda",
    **kwargs,
):
    # Parameters with defaults
    alpha0 = kwargs.get("alpha0", 1.0)
    a_exp = kwargs.get("a", math.e**2)
    beta0 = kwargs.get("beta0", 1.5)
    b_exp = kwargs.get("b", math.e)
    r = kwargs.get("r", math.exp(-1))
    c1 = kwargs.get("c1", 2.0)
    c2 = kwargs.get("c2", 2.0)
    total_iters = kwargs.get("total_iters", 7000)

    res = torch.randn_like(x0)
    for _ in range(100):
        res = AT(A(res))
        res /= torch.linalg.norm(res)
    norm = torch.sqrt(torch.linalg.norm(AT(A(res))))
    print("Estimated norm of A is", norm.item())
    tau, sigma = 1.0 / norm, 1.0 / norm

    # restart schedule
    phi = create_radial_order_schedule(schedule_length, a_exp, b_exp, c1, c2)
    ij_tuples = np.unique(phi[:, :2], axis=0)
    eps0 = torch.ones(len(ij_tuples), device=device_id)
    U, V = np.zeros(len(ij_tuples)), np.zeros(len(ij_tuples))

    x, y = x0, torch.zeros((b.shape[0], 1), device=device_id, requires_grad=False)
    vals, vals_dual = [], []

    for m in tqdm(range(schedule_length)):
        if np.sum(V) > total_iters:
            tqdm.write("Maximum number of iterations reached.")
            break

        i, j, k = phi[m]
        ij_ = find_idx_in_array(ij_tuples, [i, j])

        alpha_ = alpha0 * (a_exp**i)
        beta_ = beta0 * (b_exp**j)

        if k == 1:
            if not vals:
                eps0[ij_] = torch.linalg.norm(A(x) - b) + lam * torch.sum(torch.abs(x))
            else:
                eps0[ij_] = vals[-1] - vals_dual[-1]
        tol = (r ** U[ij_]) * eps0[ij_]
        next_tol = r * tol
        next_tol = next_tol.clamp(min=1e-5)

        if (2 * tol) > alpha_:
            delta = (2 * tol / alpha_) ** min(b_exp / beta_, 1)
        else:
            delta = (2 * tol / alpha_) ** (1 / beta_)
        delta = torch.clamp(delta, min=1e-5)

        K = torch.ceil(2 * delta * norm / next_tol).int()
        if V[ij_] + K <= k:
            tqdm.write("Start of a PDHGM run.")
            x_new, y_new, vals_alg, vals_dual_alg = pd_rLASSO(
                x,
                y,
                A,
                AT,
                b,
                tau=tau * delta,
                sigma=sigma / delta,
                num_iters=K,
                lam=lam,
                tol=next_tol,
            )
            if not vals or (vals_alg[-1] < vals[-1] + 1e-5):
                vals += [i for i in vals_alg]
                vals_dual += [i for i in vals_dual_alg]
                x = x_new
                y = y_new
            elif len(vals_alg) > 0:
                tqdm.write("The current PDHGM result is discarded.")
            V[ij_] += len(vals_alg)
            U[ij_] += 1
    return x, vals, vals_dual


def find_idx_in_array(A, target_row):
    for idx, row in enumerate(A):
        if np.all(row == target_row):
            return idx
    raise ValueError("Row not found")


def create_radial_order_schedule(t, a, b, c1, c2):
    tau, count = 1, 0  # we only create large point sets
    alpha_lim = int(abs(math.log(math.ldexp(1.0, -16)) / math.log(a)))
    beta_lim = int(abs(math.log(math.ldexp(1.0, -16)) / math.log(b)))

    # find tau
    while count < t:
        count = 0
        n1end = alpha_lim
        n2end = beta_lim
        for n1 in range(1, n1end + 1):
            if n1**c1 > tau:
                break
            for n2 in range(1, n2end + 1):
                if (n1**c1) * (n2**c2) > tau:
                    break
                for n3 in range(1, tau + 1):
                    if (n1**c1) * (n2**c2) * n3 > tau:
                        break
                    count += 2 if n1 > 1 else 1
        tau += 1
    tau -= 1

    # generate solutions
    sols_list = []
    n1end, n2end = alpha_lim, beta_lim
    for n1 in range(1, n1end + 1):
        if n1**c1 > tau:
            break
        for n2 in range(1, n2end + 1):
            if (n1**c1) * (n2**c2) > tau:
                break
            for n3 in range(1, tau + 1):
                if (n1**c1) * (n2**c2) * n3 > tau:
                    break
                sols_list.append([n1 - 1, n2 - 1, n3])
                if n1 > 1:
                    sols_list.append([1 - n1, n2 - 1, n3])

    # sort by radial schedule
    sols = torch.tensor(sols_list, dtype=torch.int64)
    schedule = (sols[:, 0].abs() + 1) * (sols[:, 1] + 1) * sols[:, 2]
    return sols[torch.argsort(schedule)]
