from pathlib import Path
import numpy as np

from NABOPB import NABOPB
from bspline_test_10d import bspline_test_10d, bspline_test_10d_fouriercoeff


# --------------------------------------------------
# settings
# --------------------------------------------------
d = 10

basis_flags = [
    "Fourier_rand",
    "Fourier_r1l",
    "Fourier_ssr1l",
    #"Fourier_mr1l",
    #"Fourier_OMP",
    #"Fourier_COSAMP",
    #"Fourier_SRLasso",
]

#gamma = {
#    "type": "sym_hc",
#    "w": 1 / 2,
#    "N": 6,
#    "sgn": "default",
#}

gamma = {
    "type": "full",
    "w": 1,
    "N": 64,
    "sgn": "default",
}

diminc = {
    "type": "default",
    "workers": 0,
}

s_values = 2 ** np.arange(3, 16)   # 2^3,...,2^15
n_repeats = 10                    

out_dir = Path("results_fourier")
out_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# storage
# --------------------------------------------------
results = {}

for basis_flag in basis_flags:
    print(f"\n===== {basis_flag} =====")

    rel_l2_fun = np.full(len(s_values), np.nan)
    coef_linf = np.full(len(s_values), np.nan)
    samples = np.full(len(s_values), np.nan)
    times = np.full(len(s_values), np.nan)

    for a, s in enumerate(s_values):
        print(f"\n--- s = {s} ---")
        
        rel_l2_runs = []
        coef_linf_runs = []
        sample_runs = []
        time_runs = []
        
        for r in range(n_repeats):
            print(f"run {r + 1}/{n_repeats}")

            result = NABOPB(
                lambda x: bspline_test_10d(x),
                d,
                basis_flag,
                gamma,
                diminc,
                int(s),
            )

            true_val, var_full = bspline_test_10d_fouriercoeff(result["index"])

            # coefficient error on detected support
            coef_err = np.linalg.norm(true_val - result["val"], np.inf)

            # relative L2 approximation error:
            # sqrt(||f||^2 - sum_{k in detected support} |true_hat_k|^2) / ||f||
            rel_l2 = np.sqrt(
                max(var_full - np.linalg.norm(true_val, 2) ** 2, 0.0)
            ) / np.sqrt(var_full)
            
            rel_l2_runs.append(rel_l2)
            coef_linf_runs.append(coef_err)
            sample_runs.append(np.sum(result["sample_sizes"]))
            time_runs.append(result["run_times"][3])

        rel_l2_fun[a] = np.median(rel_l2_runs)
        coef_linf[a] = np.median(coef_linf_runs)
        samples[a] = np.median(sample_runs)
        times[a] = np.median(time_runs)
        
        print(
            f"s={s:5d} | "
            f"relL2={rel_l2_fun[a]:.3e} | "
            f"coef∞={coef_linf[a]:.3e} | "
            f"samples={samples[a]:.0f} | "
            f"time={times[a]:.3f}s"
        )
        
        if times[a] > 3600.0:
            print(f"Stopping {basis_flag}: median time exceeded 3600s.")
            break

    results[basis_flag] = {
        "rel_l2_fun": rel_l2_fun,
        "coef_linf": coef_linf,
        "samples": samples,
        "times": times,
    }


# --------------------------------------------------
# save full data
# --------------------------------------------------
save_path = out_dir / "bspline10d_fourier_full_N64_w1_part2.npz"

np.savez(
    save_path,
    s_values=s_values,
    basis_flags=np.array(basis_flags),
    gamma_N=gamma["N"],
    gamma_w=gamma["w"],
    n_repeats=n_repeats,
    **{
        f"{basis}_rel_l2_fun": results[basis]["rel_l2_fun"]
        for basis in basis_flags
    },
    **{
        f"{basis}_coef_linf": results[basis]["coef_linf"]
        for basis in basis_flags
    },
    **{
        f"{basis}_samples": results[basis]["samples"]
        for basis in basis_flags
    },
    **{
        f"{basis}_times": results[basis]["times"]
        for basis in basis_flags
    },
)

print(f"\nSaved to {save_path}")

for quantity in ["rel_l2_fun", "coef_linf", "samples", "times"]:
    table = [s_values]

    header = ["s"]

    for basis in basis_flags:
        vals = results[basis][quantity]
        table.append(vals)
        header.append(basis)

    table = np.vstack(table).T

    csv_path = out_dir / f"{quantity}_means.csv"
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    print(f"Saved {csv_path}")
