# Koopman RMM Indices

This directory contains MATLAB code to compute realtime multivariate Madden-Julian Oscillation (RMM) indices using Koopman spectral analysis as described in the paper:

- Lintner, B. R., D. Giannakis, M. Pike, J. Slawinska (2023). Identification of the Madden–Julian Oscillation with data-driven Koopman spectral analysis. *Geophys. Res. Lett*., 50, e2023GL102743. [doi:/10.1029/2023GL102743](https://doi.org/10.1029/2023GL102743). 

The main driver script is `rmmKoopman.m`. This script has options to import RMM data into format appropriate for the NLSA code, compute the associated Koopman RMM index values, and output the results in ASCII format. 

