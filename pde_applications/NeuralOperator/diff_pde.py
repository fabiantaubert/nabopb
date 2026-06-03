from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mpmath import zeta
from fenics import *
import time
from multiprocessing import Pool
from neuralop.data.transforms.data_processors import DefaultDataProcessor

# Suppress FEniCS solver output
set_log_level(30)


def _solve_worker(y_coeffs):
    u_sol, a_expr, V = solve_diffusion_full(y_coeffs)
    a_vals, u_vals = evaluate_on_grid(u_sol, a_expr, V)
    return a_vals, u_vals


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


def solve_diffusion_full(y_coeffs, mu=2, nx=50, ny=50):
    c = float(0.9 / zeta(2))

    def k(j):
        return int(np.floor(-0.5 + np.sqrt(0.25 + 2 * j)))

    def m1(j):
        kj = k(j)
        return j - kj * (kj + 1) // 2

    def m2(j):
        return k(j) - m1(j)

    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, "P", 1)

    bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)

    class CoefficientA(UserExpression):
        def __init__(self, y, c, mu, **kwargs):
            super().__init__(**kwargs)
            self.y = y
            self.c = c
            self.mu = mu

        def eval(self, values, x_):
            x1 = x_[0]
            x2 = x_[1]
            a_val = 1.0
            for j in range(len(self.y)):
                j1 = j + 1
                freq1 = m1(j1)
                freq2 = m2(j1)
                psi_j = (
                    self.c
                    * j1 ** (-self.mu)
                    * np.cos(2 * np.pi * freq1 * x1)
                    * np.cos(2 * np.pi * freq2 * x2)
                )
                a_val += self.y[j] * psi_j
            values[0] = a_val

        def value_shape(self):
            return ()

    f = Constant(1.0)

    u = TrialFunction(V)
    v = TestFunction(V)

    a_expr = CoefficientA(y_coeffs, c, mu, degree=3)
    a_form = dot(a_expr * grad(u), grad(v)) * dx

    u_sol = Function(V)
    solve(a_form == f * v * dx, u_sol, bc)

    return u_sol, a_expr, V


def evaluate_on_grid(u_sol, a_expr, V, Gx=100):
    x = np.linspace(0, 1, Gx)
    X, Y = np.meshgrid(x, x, indexing="ij")

    u_vals = np.zeros((Gx, Gx))
    a_vals = np.zeros((Gx, Gx))

    for i in range(Gx):
        for j in range(Gx):
            p = Point(X[i, j], Y[i, j])
            u_vals[i, j] = u_sol(p)
            a_vals[i, j] = a_expr(p)

    return a_vals, u_vals


def create_diff_pde(
    n_train: int,
    n_test: int,
    n_modes: int = 20,
    n_cores: int = 1,
):
    """
    Generate and save a dataset for the parametric diffusion equation.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    n_modes : int, optional
        Number of terms in the coefficient expansion (default: 20).
    n_cores : int, optional
        Number of processes used for parallel PDE solves (default: 1).
    """

    y_train = np.random.uniform(-1, 1, (n_train, n_modes))
    y_test = np.random.uniform(-1, 1, (n_test, n_modes))

    with Pool(n_cores) as pool:
        train_results = pool.map(_solve_worker, y_train)
        test_results = pool.map(_solve_worker, y_test)

    a_train, u_train = zip(*train_results)
    a_test, u_test = zip(*test_results)

    a_train = np.stack(a_train, axis=0)
    a_train = torch.from_numpy(a_train).unsqueeze(1)
    u_train = np.stack(u_train, axis=0)
    u_train = torch.from_numpy(u_train).unsqueeze(1)
    a_test = np.stack(a_test, axis=0)
    a_test = torch.from_numpy(a_test).unsqueeze(1)
    u_test = np.stack(u_test, axis=0)
    u_test = torch.from_numpy(u_test).unsqueeze(1)

    out_dir = Path("diffusion_data") / f"train{n_train}_test{n_test}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(a_train, out_dir / "a_train.pt")
    torch.save(u_train, out_dir / "u_train.pt")
    torch.save(a_test,  out_dir / "a_test.pt")
    torch.save(u_test,  out_dir / "u_test.pt")

    print("Diffusion dataset created ✔")


def load_diff_pde(path, batch_size, test_batch_size, dtype=torch.float):
    """
    Load a previously generated dataset for the parametric diffusion equation.

    Parameters
    ----------
    path : str
        Path to the dataset folder containing
        'a_train.pt', 'u_train.pt', 'a_test.pt', and 'u_test.pt'.
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

    path = Path(path)

    a_train = torch.load(path / "a_train.pt").to(dtype)
    u_train = torch.load(path / "u_train.pt").to(dtype)
    a_test = torch.load(path / "a_test.pt").to(dtype)
    u_test = torch.load(path / "u_test.pt").to(dtype)

    train_set = TensorDictDataset(a_train, u_train)
    test_set = TensorDictDataset(a_test, u_test)

    return (
        DataLoader(train_set, batch_size, shuffle=True),
        DataLoader(test_set, test_batch_size),
        DefaultDataProcessor,
    )


if __name__ == "__main__":
    start = time.time()
    create_diff_pde(n_train=85000, n_test=1000, n_cores=96)
    end = time.time()
    print(f"Generation took {end - start:.2f} seconds.")
