# References:

*   [Unifying time evolution and optimization with MPS](https://arxiv.org/abs/1408.5056)

*   [Tangent space methods for uniform MPS](https://arxiv.org/abs/1810.07006)

*   [Variational optimization algorithms for uniform matrix product states](https://arxiv.org/abs/1701.07035)

*   [Faster Methods for Contracting Infinite 2D Tensor Networks](https://arxiv.org/abs/1711.05881)

# Finite MPS, DMRG and TDVP

1.  Complete the routines mps.jl, look for "TODO" comments in the code.
2.  Experiment with properties of the ground state, compute entanglement entropy and measure
    local expectation values
3.  Experiment with time evolution
4.  Add a two-site TDVP method, by duplicating the dmrg2sweep! method and making the necessary
    changes
5.  Experiment with different models, gapped and gapless states, ...

# Uniform MPS, VUMPS and fixed points of MPO transfer matrices

1.  Complete the routines in mps.jl, look for "TODO" comments in the code.
2.  Compute the internal energy, free energy and entropy of the Ising model
3.  Experiment with different temperatures below and above the critical point
4.  Challenge: can you also compute the magnetization?
