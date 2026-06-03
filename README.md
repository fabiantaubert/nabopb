# NABOPB

This repository provides the Python implementation of the **NABOPB algorithm** from the paper:  
**"Nonlinear approximation in bounded orthonormal product bases"**  
by Lutz Kämmerer, Daniel Potts, and Fabian Taubert.  
📄 [Read the paper here](https://doi.org/10.1007/s43670-023-00057-7)

---

## 📁 Contents

- `NABOPB.py`: Main implementation of the NABOPB algorithm.
- `NABOPB_for_PDE.py`: Modified implementation of the NABOPB algorithm for solving differential equations. Not suitable for ordinary function approximation. See examples in `pde_applications/`.
- `NABOPB_subroutines.py`: Supporting subroutines used by the main algorithm(s).
- `demo.py`: A collection of demo examples illustrating the use of the `NABOPB.py` algorithm.  
  ⚠️ **Note**: For quick tests, use small parameters (e.g., sparsity `s` and extension `Gamma.N`), or comment out unnecessary NABOPB calls.
- `bspline_test_10d.py`, `bsplinet_test_9d.py`, `cardinal_bspline.py`: Implementations of test functions based on splines and the cardinal B-spline. Used for tests in `demo.py` and `pde_applications/`.
- `r1lfft/`, `mr1lfft/`, `cmr1lfft/`: Python implementations for generating rank-1 lattices and performing fast transforms on them. Based on original MATLAB code by Lutz Kämmerer.
- `sparse_recovery/`: Python implementations of OMP, CoSaMP and SR-LASSO. Based on the [SparseRecovery Repository](https://github.com/Zeppo1994/SparseRecovery) by Sebastian Neumayer.
- `pde_applications/`: Applications of the NABOPB algorithm to differential equations. See below for more details.

---

## ▶️ Getting Started

To run all demo examples, use:

```bash
python demo.py
```

Alternatively, you can run individual cells in `demo.py` using your preferred IDE.

---

# PDE Applications

The `pde_applications/` subfolder contains implementations of numerical experiments that apply the **NABOPB algorithm** for approximating solution operators of differential equations, as presented in the following papers:  
**"An approach to discrete operator learning based on sparse high-dimensional approximation"**  
by Daniel Potts and Fabian Taubert.  
📄 [Read the paper here](https://doi.org/10.1553/etna_vol63s468)

**"Learning solution operators of PDEs with sparse approximation methods"**  
by Sebastian Neumayer, Daniel Potts and Fabian Taubert.  
📄 [Read the preprint here](https://arxiv.org/pdf/)

---

## 📁 Contents

Scripts are named according to the differential equation they address. For example, `demo_heat.py` corresponds to the heat equation as used in the corresponding examples in the two papers.

- `demo_*.py`: Scripts to compute and save the approximation using a differential equation solver as a function handle.
- `testing_*.py`: Scripts to load the results and compute/plot errors.
- `demo_pois1d_highdim.py`: High-dimensional extension example (Example 3.2 in "An approach to discrete operator learning based on sparse high-dimensional approximation").
- `NeuralOperator/`: Subfolder for the TFNO comparison. Contains scripts for generating data and Notebooks for training and evaluating TFNOs for our examples. Based on the [NeuralOperator Repository](https://github.com/neuraloperator/neuraloperator).

---

## ▶️ Getting Started

To run a demo script:

```bash
python demo_xxx.py
```

This saves results (by default) in a subfolder such as `results_xxx/` as `.pickle` files.

Then, to evaluate the results:

```bash
python testing_xxx.py
```

⚠️ **Note**: The demo scripts for the heat equation, Burgers' equation, and the parametric diffusion equation use `s = 1000`, `N = 64`, and the `OMP+` approach by default, corresponding to the experiments in "Learning solution operators of PDEs with sparse approximation methods". The remaining demo scripts use `s = 100`, `N = 16`, and the `cMR1L` approach by default, corresponding to the experiments in "An approach to discrete operator learning based on sparse high-dimensional approximation". These parameters can be adjusted directly in the scripts. Saved filenames reflect the chosen parameter settings.
