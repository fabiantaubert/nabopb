# NABOPB

This repository provides the Python implementation of the **NABOPB algorithm** from the paper:  
**"Nonlinear approximation in bounded orthonormal product bases"**  
by Lutz Kämmerer, Daniel Potts, and Fabian Taubert.  
📄 [Read the paper here](https://doi.org/10.1007/s43670-023-00057-7)

---

## 📁 Contents

- `NABOPB.py`: Main implementation of the NABOPB algorithm.
- `NABOPB_subroutines.py`: Supporting subroutines used by the main algorithm.
- `demo.py`: A collection of demo examples illustrating the use of the NABOPB algorithm.  
  ⚠️ **Note**: For quick tests, use small parameters (e.g., sparsity `s` and extension `Gamma.N`), or comment out unnecessary NABOPB calls.
- `bspline_test_10d.py`, `bsplinet_test_9d.py`, `cardinal_bspline.py`: Implementations of test functions based on splines and the cardinal B-spline. Used for tests in `demo.py` and `pde_applications/`.
- `r1lfft/`, `mr1lfft/`, `cmr1lfft/`: Python implementations for generating rank-1 lattices and performing fast transforms on them. Based on original MATLAB code by Lutz Kämmerer.
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

The `pde_applications/` subfolder contains implementations of numerical experiments that apply the **NABOPB algorithm** to solve differential equations, as presented in the paper:  
**"An approach to discrete operator learning based on sparse high-dimensional approximation"**  
by Daniel Potts and Fabian Taubert.  
📄 [Read the preprint here](https://arxiv.org/pdf/2406.03973)

---

## 📁 Contents

Scripts are named according to the differential equation they address. For example, `demo_pois_2d.py` corresponds to the two-dimensional Poisson equation from Section 3.3 of the paper.

- `demo_*.py`: Scripts to compute and save the approximation using a differential equation solver as a function handle.
- `testing_*.py`: Scripts to load the results and compute/plot errors.
- `demo_pois1d_highdim.py`: High-dimensional extension example (Example 3.2 in the paper).

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

⚠️ **Note**: All demo scripts use `s = 100` and `N = 16` by default. You can adjust these parameters in the scripts (e.g., `s = 1000`, `N = 64`, as used in the paper). The saved filenames will reflect the parameter choices for easy identification.
