from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.integrate import solve_ivp
import time
from multiprocessing import Pool
from neuralop.data.transforms.data_processors import DefaultDataProcessor


def _solve_worker(args):
    x_grid, t_grid, a_coeffs = args
    return solve_heat_full(x_grid, t_grid, a_coeffs)


class TensorDictDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
        }


def solve_heat_full(
    x_grid,
    t_grid,
    a_coeffs,
    alpha=0.25,
    nx=1000,
    rtol=1e-8,
    atol=1e-10,
):
    x_vals = np.linspace(0, 1, nx)
    dx = x_vals[1] - x_vals[0]

    u0 = np.zeros(nx)
    for l, a_l in enumerate(a_coeffs, start=1):
        u0 += a_l * np.sin(l * np.pi * x_vals)

    u0[0] = u0[-1] = 0.0

    def heat_rhs(t, u):
        dudt = np.zeros_like(u)
        dudt[1:-1] = alpha**2 * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        return dudt

    sol = solve_ivp(
        heat_rhs,
        (0.0, t_grid[-1]),
        u0,
        t_eval=t_grid,
        method="Radau",
        rtol=rtol,
        atol=atol,
    )

    u_out = np.zeros((len(t_grid), len(x_grid)))
    for k in range(len(t_grid)):
        u_out[k] = np.interp(x_grid, x_vals, sol.y[:, k])

    return u_out  # shape (Gt, Gx)


def create_heat_1dtime(
    n_train: int,
    n_test: int,
    n_modes: int = 9,
    Gx: int = 100,
    Gt: int = 100,
    n_cores: int = 1,
):
    """
    Generate and save a dataset for the 1D heat equation.

    The resulting tensors are stored as PyTorch files in
    'heat_data/train{n_train}_test{n_test}/'.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    n_modes : int, optional
        Number of sine modes used in the initial condition (default: 9).
    Gx : int, optional
        Number of spatial grid points (default: 100).
    Gt : int, optional
        Number of temporal grid points (default: 100).
    n_cores : int, optional
        Number of processes used for parallel PDE solves (default: 1).
    """

    x_grid = np.linspace(0, 1, Gx)
    t_grid = np.linspace(0, 1, Gt)

    a_coeffs_train = np.sin(np.pi * (np.random.rand(n_train, n_modes) - 1 / 2))
    a_coeffs_test = np.sin(np.pi * (np.random.rand(n_test, n_modes) - 1 / 2))

    u0_train = np.zeros((n_train, 1, Gx))
    u0_test = np.zeros((n_test, 1, Gx))

    for l in range(n_modes):
        u0_train[:, 0, :] += a_coeffs_train[:, l][:, None] * np.sin(
            (l + 1) * np.pi * x_grid
        )
        u0_test[:, 0, :] += a_coeffs_test[:, l][:, None] * np.sin(
            (l + 1) * np.pi * x_grid
        )

    u0_train[:, 0, 0] = 0.0
    u0_train[:, 0, -1] = 0.0
    u0_test[:, 0, 0] = 0.0
    u0_test[:, 0, -1] = 0.0

    u_train = np.zeros((n_train, 1, Gt, Gx))
    u_test = np.zeros((n_test, 1, Gt, Gx))

    with Pool(processes=n_cores) as pool:
        train_results = pool.map(
            _solve_worker,
            [(x_grid, t_grid, a_coeffs_train[j]) for j in range(n_train)],
        )
        test_results = pool.map(
            _solve_worker,
            [(x_grid, t_grid, a_coeffs_test[j]) for j in range(n_test)],
        )

    u_train[:, 0, :, :] = np.stack(train_results, axis=0)
    u_test[:, 0, :, :] = np.stack(test_results, axis=0)

    u0_train = torch.from_numpy(u0_train)
    u_train = torch.from_numpy(u_train)
    u0_test = torch.from_numpy(u0_test)
    u_test = torch.from_numpy(u_test)

    out_dir = Path("heat_data") / f"train{n_train}_test{n_test}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(u0_train, out_dir / "u0_train.pt")
    torch.save(u_train, out_dir / "u_train.pt")
    torch.save(u0_test, out_dir / "u0_test.pt")
    torch.save(u_test, out_dir / "u_test.pt")

    print("Data creation complete ✔")


def load_heat_1dtime(
    path: str,
    batch_size: int,
    test_batch_size: int,
    dtype=torch.float,
):
    """
    Load a previously generated dataset for the 1D heat equation.

    Parameters
    ----------
    path : str
        Path to the dataset folder containing
        'u0_train.pt', 'u_train.pt', 'u0_test.pt', and 'u_test.pt'.
    batch_size : int
        Batch size used for the training DataLoader.
    test_batch_size : int
        Batch size used for the test DataLoader.
    dtype : torch.dtype, optional
        Data type to which the loaded tensors are converted (default: torch.float).

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader providing batches of training samples.
    test_loader : torch.utils.data.DataLoader
        DataLoader providing batches of test samples.
    data_processor : type
        DefaultDataProcessor class from NeuralOperator, to be instantiated externally.
    """

    folder = Path(path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist")

    u0_train = torch.load(folder / "u0_train.pt").to(dtype)
    u_train = torch.load(folder / "u_train.pt").to(dtype)
    u0_test = torch.load(folder / "u0_test.pt").to(dtype)
    u_test = torch.load(folder / "u_test.pt").to(dtype)

    Gt = u_train.shape[2]

    u0_train = u0_train.unsqueeze(2).repeat(1, 1, Gt, 1)
    u0_test = u0_test.unsqueeze(2).repeat(1, 1, Gt, 1)

    train_set = TensorDictDataset(u0_train, u_train)
    test_set = TensorDictDataset(u0_test, u_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, DefaultDataProcessor


if __name__ == "__main__":
    start = time.time()
    create_heat_1dtime(n_train=40000, n_test=500, n_cores=96)
    end = time.time()
    print(f"Generation took {end - start:.2f} seconds.")
