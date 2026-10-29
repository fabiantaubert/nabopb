# NABOPB

This repository provides the Python implementation of the **NABOPB algorithm**, originally introduced in the paper:  
**"Nonlinear approximation in bounded orthonormal product bases"**  
by Lutz Kämmerer, Daniel Potts, and Fabian Taubert.  
📄 [Read the paper here](https://doi.org/10.1007/s43670-023-00057-7)

The repository has since been extended and now also serves as the accompanying software for Fabian Taubert's dissertation **"A Dimension-Incremental Framework for Sparse High-Dimensional Function Approximation and Operator Learning"**. The dissertation provides a more comprehensive treatment of the underlying framework, its extensions, and numerical applications. A link will be added once it is publicly available.

---

## 📁 Contents

- `NABOPB.py`: Main implementation of the NABOPB algorithm.
- `NABOPB_for_PDE.py`: Modified implementation for the sparse approximation of PDE solution operators, exploiting additional spatial and temporal solution values to reduce the number of required differential equation solves. See examples in `pde_applications/`.
- `NABOPB_subroutines.py`: Supporting subroutines used by the main algorithm(s).
- `tutorial_nabopb.ipynb`: Interactive introduction explaining the dimension-incremental algorithm and its main steps.
- `tutorial_usage.ipynb`: Practical tutorial on using NABOPB, choosing coefficient reconstruction methods, and interpreting the results.
- `tutorial_pde.ipynb`: Tutorial demonstrating the application of NABOPB and NABOPB_for_PDE to Poisson and heat equations.
- `demo.py`: A collection of demo examples illustrating the use of the `NABOPB.py` algorithm.
- `experiments_fourier_10d.py`, `experiments_chebyshev_9d.py`: Scripts for the numerical experiments with 10-dimensional periodic and 9-dimensional non-periodic test functions, respectively, presented in Chapter 3 of the dissertation.
- `requirements.txt`: List of Python packages required by the main NABOPB implementation.
- `bspline_test_10d.py`, `bsplinet_test_9d.py`, `cardinal_bspline.py`: Implementations of test functions based on splines and the cardinal B-spline. Used in the demos and numerical experiments.
- `r1lfft/`, `mr1lfft/`, `cmr1lfft/`: Python implementations for generating rank-1 lattices and performing fast transforms on them. Based on original MATLAB code by Lutz Kämmerer.
- `sparse_recovery/`: Python implementations of OMP, CoSaMP and SR-LASSO. Based on the [SparseRecovery Repository](https://github.com/Zeppo1994/SparseRecovery) by Sebastian Neumayer.
- `pde_applications/`: Applications of the NABOPB algorithm to differential equations. See below for more details.

---

## ▶️ Getting Started

**Windows users:** Native Windows is currently not supported due to compatibility issues with the `pykeops` dependency, which also requires a C++ compiler. We recommend using the Windows Subsystem for Linux (WSL) with Ubuntu. Install WSL by running `wsl --install` in PowerShell as administrator. Then, inside Ubuntu, install the required development tools:

```bash
sudo apt update
sudo apt install build-essential python3-venv python3-pip
```

Afterwards, follow the installation instructions below from within WSL. VS Code can connect to the WSL environment using its [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl).

**Note:** This setup has not been tested by the authors.

First, install the required Python packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

For a guided introduction to NABOPB, we recommend the following Jupyter notebooks:

1. `tutorial_nabopb.ipynb`: Introduction to the algorithm and its dimension-incremental procedure.
2. `tutorial_usage.ipynb`: Practical examples of different bases, coefficient reconstruction methods, and approximation results.
3. `tutorial_pde.ipynb`: Applications to Poisson and heat equations, including the PDE-specific extension of NABOPB.

To run the notebooks, install Jupyter if necessary:

```bash
python -m pip install notebook ipykernel
python -m notebook
```

Alternatively, `demo.py` contains additional examples that can be executed individually in a Python IDE or together using:

```bash
python demo.py
```

Some examples and notebook cells are computationally expensive and may require substantial runtime.

---

# PDE Applications

The `pde_applications/` subfolder contains implementations of numerical experiments that apply the **NABOPB algorithm** to the approximation of solution operators of differential equations, as presented in the following papers:

**"An approach to discrete operator learning based on sparse high-dimensional approximation"**  
by Daniel Potts and Fabian Taubert.  
📄 [Read the paper here](https://doi.org/10.1553/etna_vol63s468)

**"Learning solution operators of PDEs with sparse approximation methods"**  
by Sebastian Neumayer, Daniel Potts, and Fabian Taubert.  
📄 [Read the preprint here](https://arxiv.org/abs/2606.06046)

---

## 📁 Contents

Scripts are named according to the differential equation they address. For example, `demo_heat_1d.py` corresponds to the heat equation considered in the papers.

- `demo_*.py`: Scripts for computing and saving sparse approximations using differential equation solvers as function handles.
- `testing_*.py`: Scripts for loading the saved approximations and computing or plotting approximation errors.
- `demo_pois_highdim.py`: High-dimensional extension example (Example 3.2 in "An approach to discrete operator learning based on sparse high-dimensional approximation").
- `NeuralOperator/`: Subfolder for comparisons with Tensorized Fourier Neural Operators (TFNOs). Contains scripts for generating training data and notebooks for training and evaluating TFNOs. Based on the [NeuralOperator Repository](https://github.com/neuraloperator/neuraloperator).

---

## ▶️ Getting Started

⚠️ **Dependencies**: Depending on the example, additional packages such as FEniCS, Matplotlib, mpmath, or NeuralOperator may be required. These packages are not included in the main `requirements.txt`.

To run an example from the repository root, execute the corresponding demo script, for example:

```bash
python pde_applications/demo_heat_1d.py
```

This saves the computed approximation as a `.pickle` file in a corresponding results folder, such as `pde_applications/results_heat_1d/`.

The saved results can then be evaluated using the corresponding testing script:

```bash
python pde_applications/testing_heat_1d.py
```

⚠️ **Note**: Most PDE demo scripts currently use `s = 1000` and `N = 64` by default. The coefficient reconstruction method depends on the example and includes `Cheby_OMP+`, `Cheby_OMP`, and `Cheby_mr1l`. The separate high-dimensional Poisson example has its own settings. Parameters can be adjusted directly in the scripts, and saved filenames reflect the chosen settings. The numerical experiments can require substantial computational resources.
