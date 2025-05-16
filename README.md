# NABOPB

This repository provides the Python implementation of the **NABOPB algorithm** from the paper:  
**"Nonlinear approximation in bounded orthonormal product bases"**  
by Lutz Kämmerer, Daniel Potts, and Fabian Taubert.  
📄 [Read the paper here](https://doi.org/10.1007/s43670-023-00057-7)

---

## 📁 Contents

- *NABOPB.py*: Main implementation of the NABOPB algorithm.
- *NABOPB_subroutines.py*: Supporting subroutines required by the main algorithm.
- *demo.py*: A collection of demo examples demonstrating the use of the NABOPB algorithm.  
  ⚠️ **Note**: For quick tests, use small parameters (e.g., sparsity `s` and extension `Gamma.N`), or comment out unnecessary NABOPB calls.
- *bspline_test_10d.py*, *bsplinet_test_9d.py*, *cardinal_bspline.py*: Python implementations of test functions made from splines and the cardinal B-spline. Used for tests in *demo.py* and *pde_applications/*.
- *r1lfft/*, *mr1lfft/*, *cmr1lfft/*: Python implementations for generating rank-1 lattices and performing fast transforms on them. These implementations are based on original MATLAB code by Lutz Kämmerer.
- *pde_applications/*: Implementations for the application of the NABOPB algorithm to differential equation examples. See below for details.

---

## ▶️ Getting Started

Run a single cell of *demo.py* using an IDE of your choice. To run all the demo examples, use

```bash
python demo.py
