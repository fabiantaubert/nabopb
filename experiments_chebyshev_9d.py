from pathlib import Path
import numpy as np

from NABOPB import NABOPB
from bsplinet_test_9d import bsplinet_test_9d, bsplinet_test_9d_chat


# --------------------------------------------------
# settings
# --------------------------------------------------
d = 9

basis_flags = [
 #   "Cheby_rand",
 #   "Cheby_mr1l",
 #   "Cheby_OMP",
 #   "Cheby_COSAMP",
    "Cheby_SRLasso",
]

gamma = {
    "type": "sym_hc",
    "w": 1 / 2,
    "N": 8,
    "sgn": "non-negative",
}

diminc = {
    "type": "default",
    "workers": 0,
}

s_values = 2 ** np.arange(3, 16)   # 2^3, ..., 2^15
n_repeats = 3
max_median_time = 3600.0

out_dir = Path("results_cheby")
out_dir.mkdir(parents=True, exist_ok=True)


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
                lambda x: bsplinet_test_9d(x),
                d,
                basis_flag,
                gamma,
                diminc,
                int(s),
            )

            # Convert the recovered coefficients to the normalization
            # used by bsplinet_test_9d_chat.
            normalization = np.sqrt(
                np.prod(
                    2 ** (result["index"] != 0),
                    axis=1,
                )
            )
            recovered_val = result["val"] * normalization

            # True coefficients on the detected index set and full norm squared
            true_val, norm_full_sq = bsplinet_test_9d_chat(result["index"])

            # Coefficient approximation error on the detected support
            coef_err = np.linalg.norm(
                true_val - recovered_val,
                ord=np.inf,
            )

            # Relative L2 function approximation error resulting from
            # indices missing from the detected support.
            # Convert the exact coefficients to the orthonormal Chebyshev
            # normalization before applying Parseval.
            true_val_orth = true_val / normalization
            
            captured_energy = np.linalg.norm(true_val_orth, 2) ** 2
            
            rel_l2 = np.sqrt(
                max(norm_full_sq - captured_energy, 0.0)
            ) / np.sqrt(norm_full_sq)

            rel_l2_runs.append(rel_l2)
            coef_linf_runs.append(coef_err)
            sample_runs.append(np.sum(result["sample_sizes"]))
            time_runs.append(result["run_times"][3])

        # Store only the median over all repetitions
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

        if times[a] > max_median_time:
            print(
                f"Stopping {basis_flag}: median time exceeded "
                f"{max_median_time:.0f}s."
            )
            break

    results[basis_flag] = {
        "rel_l2_fun": rel_l2_fun,
        "coef_linf": coef_linf,
        "samples": samples,
        "times": times,
    }

# --------------------------------------------------
# save complete median data
# --------------------------------------------------
save_path = out_dir / "bsplinet9d_cheby_symhc_N8_w12_LASSO.npz"

np.savez(
    save_path,
    s_values=s_values,
    basis_flags=np.array(basis_flags),
    gamma_type=gamma["type"],
    gamma_N=gamma["N"],
    gamma_w=gamma["w"],
    n_repeats=n_repeats,
    max_median_time=max_median_time,
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


# --------------------------------------------------
# CSV files for PGFPlots
# --------------------------------------------------
for quantity in ["rel_l2_fun", "coef_linf", "samples", "times"]:
    table = [s_values]
    header = ["s"]

    for basis in basis_flags:
        table.append(results[basis][quantity])
        header.append(basis)

    table = np.vstack(table).T

    csv_path = out_dir / f"cheby_{quantity}_medians_LASSO.csv"

    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    print(f"Saved {csv_path}")
