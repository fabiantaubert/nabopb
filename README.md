# NABOPB

The Python code for the NABOPB algorithm presented in the paper "Nonlinear approximation in bounded orthonormal product bases" by Lutz Kämmerer, Daniel Potts and Fabian Taubert.

The main algorithm is given in NABOPB.py together with the necessary subroutines in NABOPB_subroutines.py

demo.py contains several demo examples for the application of the algorithm. It is highly recommended to use small parameters (sparsity s and extension Gamma.N) or to comment the unwanted executions of the algorithm beforehand.

r1lfft, mr1lfft and cmr1lfft contain the Python codes for the generation of the rank-1 lattices as well as the fast transforms on these lattices. The codes mainly originate from MATLAB codes by Lutz Kämmerer.
