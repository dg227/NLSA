# NLSA

This project provides a Matlab implementation of nonlinear Laplacian spectral analysis (NLSA) and related kernel algorithms for feature extraction and prediction of observables of dynamical systems. 

## Usage

1. Clone down the project repository (`git clone https://github.com/dg227/NLSA`).
2. Launch Matlab, `cd` into the project's directory, and add `/nlsa` to the Matlab search path. This can be done by executing the Matlab command:
```matlab
addpath(genpath('nlsa'))
``` 

## Examples

- Rectification of variable speed oscillator. 
```matlab
/examples/circle/demoCircle.m
``` 
- Extraction of an approximately cyclical observable of the Lorenz 63 (L53) chaotic system.
- Kernel analog forecasting of the L63 state vector components.


## Acknowledgement 

Research funded by the [National Science Foundation](https://nsf.gov) (grants DMS-1521775, 1842538, DMS-1854383) and [Office of Naval Research](https://onr.navy.mil) (grants N00014-14-1-0150, N00014-16-1-2649, N00014-19-1-2421).

<div align="center"><img src="pages/img/logoNSF.jpg" alt="NSF logo" height="70" hspace="10"><img src="pages/img/logoONR.png" alt="ONR logo" height="70" hspace="10"></div>
