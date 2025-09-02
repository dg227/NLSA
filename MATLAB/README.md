# MATLAB NLSA code

This directory contains a MATLAB implementation of nonlinear Laplacian spectral analysis (NLSA) and related operator methods for feature extraction and prediction of observables of dynamical systems. 

## Usage

1. Clone down the project repository:
```shell
git clone https://github.com/dg227/NLSA
```
2. Add directory `NLSA/MATLAB/nlsa` to the MATLAB search path. This can be done by launching MATLAB, `cd`'ing to directory `NLSA/MATLAB`, and executing the command:
```matlab
addpath(genpath('nlsa'))
```

## Examples

- Rectification of variable-speed periodic oscillator using Koopman eigenfunctions: 
```shell
NLSA/MATLAB/examples/circle/demoKoopman.m
```
- Extraction of an approximately cyclical observable of the Lorenz 63 (L63) chaotic system using kernel integral operators with delays:
```shell
NLSA/MATLAB/examples/circle/demoNLSA.m
``` 
- Kernel analog forecasting of the L63 state vector components:
```shell
NLSA/MATLAB/examples/l63/demoKAF.m
```

## Implementation

NLSA implements a MATLAB class ``nlsaModel`` which encodes the attributes of the machine learning procedure to be carried out. This includes:
- Specification of training and test data.
- Delay-coordinate embedding.
- Pairwise distance functions.
- Density estimation for variable-bandwidth kernels.
- Symmetric and non-symmetric Markov kernels.
- Koopman operators.
- Projection and reconstruction of target data.   
- Nystrom out-of-sample extension.

Each of the elements above are implemented as MATLAB classes. See ``NLSA/MATLAB/nlsa/classes`` for further information and basic documentation.

Results from each stage of the computation are written on disk in a directory tree with (near-) unique names based on the `nlsaModel` parameters. 


## Known issues

- In Windows environments, errors can occur due to long file/directory names. 
