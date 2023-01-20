# Quantum Mechanical Data Assimilation

This directory contains MATLAB and Python code to perform quantum mechanical data assimilation (QMDA) and plot some of the figures in the paper:

- Freeman, D., D. Giannakis, B. Mintz, A. Ourmazd, J. Slawinska (2023). Data assimilation in operator algebras. *Proc. Natl. Acad. Sci*. [doi:10.1073/pnas.2211115](https://dx.doi.org/10.1073/pnas.2211115). 

The code organized in the following two subdirectories:
- `enso_qmda`: El Nino Southern Oscillation (ENSO) from the Community Climate System Model version 4 (CCSM4). 
- `l96Multiscale_qmda`: Lorenz L96 (L66) multiscale system.

For the MATLAB portion of the code, the main functions implementing Koopman operator approximation and QMDA can be found in `/nlsa/utils/forecasting`. Additional kernel functions used for QMDA are in `/nlsa/utils/kernels`. The modules associated with the Python portion of the code can be found in `/Python/src/nlsa`. 

The code was tested on MATLAB R2022a and Python 3.10 running on macOS and Ubuntu Linux.

## ENSO experiments

The ENSO experiments can be run using the MATLAB script `./enso_qmda/ensoQMDA.m`, which reproduces Figs. 4 and 5 of the paper. The script will import the CCSM4 training and test data, compute data-driven basis functions using NLSA, and run the QMDA algorithm. The script requires that model output from CCSM4 in NetCDF format (typically downloaded from a public repository; e.g., the `b40.1850` control run available at the [Earth System Grid repository](https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.joc.b40.1850.track1.1deg.006.html)) is placed in the directory `rootDataPath` assigned in the file `./enso_qmda/importData.m`.

## L96 multiscale experiments

The L96 multiscale experiments can be run using the MATLAB script `./l96Multiscale_qmda/l96MultiscaleQMDA.m`, which reproduces Figs. 2 and 3 of the paper. The script will generate the L96 multiscale data, compute data-driven basis functions using NLSA, and run the QMDA algorithm. The quantum circuit results in Figs. 6 and 7 can be reproduced using the Jupyter notebook `./l96Multiscale_qmda/l96MultiscaleQMDA.ipynb`. The notebook uses output from the MATLAB portion of the code. In addition, it requires that the Python NLSA package in `/Python` is installed; see `/Python/README.md` for additional information.

## Issues:

- On Windows machines errors may occur due to long directory names generated
  by the code. 
