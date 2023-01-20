# Spectral analysis of climate dynamics with operator-theoretic approaches

This directory contains scripts to compute Koopman eigenfunctions and plot some of the figures in the paper:
- G. Froyland, D. Giannakis, B. Lintner, M. Pike, J. Slawinska (2021). Spectral analysis of climate dynamics with operator-theoretic approaches. *Nat. Commun.* 12, 6570. [doi:10.1038/s41467-021-26357-x](https://doi.org/10.1038/s41467-021-26357-x)

The scripts are organized in the following three subdirectories:
- `circle`:  Variable-speed periodic oscillator on the circle.
- `l63`:     Lorenz 63 dynamical system.
- `enso`:    El Nino lifecycle from Indo-Pacific sea surface temperature data.

Each subdirectory contains a driver script, `demoKoopman.m`, along with various auxiliary functions to generate the data and set the data analysis parameters. 

Running demoKoopman will generate/import the data, perform eigendecomposition of the Koopman generator, and plot figures/movies.

Data output is written in a directory tree contained in subdirectory `data`.

Figures and movies are written in subdirectory `figs`. 

Requirements:
- The ENSO examples require the Financial Toolbox for calculation of calendar date ranges.
- Some plots use the Image Processing Toolbox for color selection.

The code was tested on MATLAB R2021a running on macOS and Ubuntu Linux.

Issues:
- On Windows machines errors may occur due to long directory names generated
  by the code. 
