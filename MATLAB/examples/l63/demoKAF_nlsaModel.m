function [ model, In, Out ] = demoKAF_nlsaModel( experiment )
% DEMOKAF_NLSAMODEL Construct NLSA model for kernel analog forecasting of the
% of Lorenz 63 system.
%
% Input arguments:
%
% experiment: A string identifier for the data analysis experiment. 
%
% Output arguments:
%
% model: Constructed model, in the form of an nlsaModel object.  
% In:    Data structure with in-sample model parameters. 
% Out:   Data structure with out-of-sample model parameters (optional). 
%
% nT is the number of extra samples we include in order to have availbale the
% full evolution of the response variable over the desired forecast lead times.
%
% This function creates the data structures In and Out, which are then passed 
% to function l63NLSAModel to build the model.
%
% Modified 2021/02/27   
%
if nargin == 0
    experiment = '6.4k_dt0.01_idxX1_2_3_nEL0';
end

switch experiment

    % 6400 samples, sampling interval 0.01, no delay embedding, full obs. 
    case '6.4k_dt0.01_idxX1_2_3_nEL0'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 6400;         % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Out-of-sample dataset parameters
        Out.dt         = 0.01;         % sampling interval
        Out.Res.beta   = 8/3;          % L63 parameter beta
        Out.Res.rho    = 28;           % L63 parameter rho
        Out.Res.sigma  = 10;           % L63 parameter sigma
        Out.Res.nSProd = 6400;         % number of "production" samples
        Out.Res.nSSpin = 64000;        % spinup samples
        Out.Res.x0     = [ 0.2 1.2 1.25 ]; % initial conditions
        Out.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        Out.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 1;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 500;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nParNN     = 2;          % parallel workers for nearest neighbors
        In.nParE      = 0;          % workers for delay-embedding sums
        In.nN         = 0;          % nearest neighbors for pairwise distances
        In.lDist      = 'l2';       % local distance
        In.tol        = 0;          % 0 distance threshold (for cone kernel)
        In.zeta       = 0;          % cone kernel parameter 
        In.coneAlpha  = 0;          % velocity exponent in cone kernel
        In.nNS        = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = .5;        % diffusion maps normalization 
        In.nPhi        = 1001;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

        % NLSA parameters, out-of-sample data
        Out.Res.nB    = 1;
        Out.Res.nBRec = 1;

    % 6400 samples, sampling interval 0.01, no delay embedding, full obs., 
    % RBF kernel. 
    case '6.4k_dt0.01_idxX1_2_3_nEL0_rbf'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 6400;         % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Out-of-sample dataset parameters
        Out.dt         = 0.01;         % sampling interval
        Out.Res.beta   = 8/3;          % L63 parameter beta
        Out.Res.rho    = 28;           % L63 parameter rho
        Out.Res.sigma  = 10;           % L63 parameter sigma
        Out.Res.nSProd = 6400;         % number of "production" samples
        Out.Res.nSSpin = 64000;        % spinup samples
        Out.Res.x0     = [ 0.2 1.2 1.25 ]; % initial conditions
        Out.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        Out.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 1;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 500;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nParNN     = 0;          % parallel workers for nearest neighbors
        In.nParE      = 0;          % workers for delay-embedding sums
        In.nN         = 0;          % nearest neighbors for pairwise distances
        In.lDist      = 'l2';       % local distance
        In.tol        = 0;          % 0 distance threshold (for cone kernel)
        In.zeta       = 0;          % cone kernel parameter 
        In.coneAlpha  = 0;          % velocity exponent in cone kernel
        In.nNS        = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb';    % diffusion operator type
        In.epsilon     = 2;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = 0;         % diffusion maps normalization 
        In.beta        = 0;         % diffusion maps normalization (Markov)
        In.nPhi        = 1001;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, out-of-sample data
        Out.Res.nB    = 1;
        Out.Res.nBRec = 1;

    % 6400 samples, sampling interval 0.01, no delay embedding, x1 only is
    % observed. 
    case '6.4k_dt0.01_idxX1_nEL0'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 6400;         % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Out-of-sample dataset parameters
        Out.dt         = 0.01;         % sampling interval
        Out.Res.beta   = 8/3;          % L63 parameter beta
        Out.Res.rho    = 28;           % L63 parameter rho
        Out.Res.sigma  = 10;           % L63 parameter sigma
        Out.Res.nSProd = 6400;         % number of "production" samples
        Out.Res.nSSpin = 64000;        % spinup samples
        Out.Res.x0     = [ 0.2 1.2 1.25 ]; % initial conditions
        Out.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        Out.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 1;       % observed state vector components 
        In.Src.idxE    = 1 : 1;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 500;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nParNN     = 2;          % parallel workers for nearest neighbors
        In.nParE      = 0;          % workers for delay-embedding sums
        In.nN         = 0;          % nearest neighbors for pairwise distances
        In.lDist      = 'l2';       % local distance
        In.tol        = 0;          % 0 distance threshold (for cone kernel)
        In.zeta       = 0;          % cone kernel parameter 
        In.coneAlpha  = 0;          % velocity exponent in cone kernel
        In.nNS        = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = .5;        % diffusion maps normalization 
        In.nPhi        = 1001;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 1;             % manifold dimension
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

        % NLSA parameters, out-of-sample data
        Out.Res.nB    = 1;
        Out.Res.nBRec = 1;

    % 6400 samples, sampling interval 0.01, 15 delays, full obs
    case '6.4k_dt0.01_idxX1_2_3_nEL15'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 6400;         % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Out-of-sample dataset parameters
        Out.dt         = 0.01;         % sampling interval
        Out.Res.beta   = 8/3;          % L63 parameter beta
        Out.Res.rho    = 28;           % L63 parameter rho
        Out.Res.sigma  = 10;           % L63 parameter sigma
        Out.Res.nSProd = 6400;         % number of "production" samples
        Out.Res.nSSpin = 64000;        % spinup samples
        Out.Res.x0     = [ 0.2 1.2 1.25 ]; % initial conditions
        Out.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        Out.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 16;      % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 500;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nParNN     = 2;          % parallel workers for nearest neighbors
        In.nParE      = 2;          % workers for delay-embedding sums
        In.nN         = 0;          % nearest neighbors for pairwise distances
        In.lDist      = 'l2';       % local distance
        In.tol        = 0;          % 0 distance threshold (for cone kernel)
        In.zeta       = 0;          % cone kernel parameter 
        In.coneAlpha  = 0;          % velocity exponent in cone kernel
        In.nNS        = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = .5;        % diffusion maps normalization 
        In.nPhi        = 1001;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;            % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

        % NLSA parameters, out-of-sample data
        Out.Res.nB    = 1;
        Out.Res.nBRec = 1;

    % 6400 samples, sampling interval 0.01, 15 delays, full obs., RBF kernel
    case '6.4k_dt0.01_idxX1_2_3_nEL15_rbf'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 6400;         % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Out-of-sample dataset parameters
        Out.dt         = 0.01;         % sampling interval
        Out.Res.beta   = 8/3;          % L63 parameter beta
        Out.Res.rho    = 28;           % L63 parameter rho
        Out.Res.sigma  = 10;           % L63 parameter sigma
        Out.Res.nSProd = 6400;         % number of "production" samples
        Out.Res.nSSpin = 64000;        % spinup samples
        Out.Res.x0     = [ 0.2 1.2 1.25 ]; % initial conditions
        Out.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        Out.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 16;      % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 500;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nParNN     = 2;          % parallel workers for nearest neighbors
        In.nParE      = 2;          % workers for delay-embedding sums
        In.nN         = 0;          % nearest neighbors for pairwise distances
        In.lDist      = 'l2';       % local distance
        In.tol        = 0;          % 0 distance threshold (for cone kernel)
        In.zeta       = 0;          % cone kernel parameter 
        In.coneAlpha  = 0;          % velocity exponent in cone kernel
        In.nNS        = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb';    % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = .5;        % diffusion maps normalization 
        In.nPhi        = 1001;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;            % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

        % NLSA parameters, out-of-sample data
        Out.Res.nB    = 1;
        Out.Res.nBRec = 1;


    % 6400 samples, sampling interval 0.01, 15 delays, x1 only is observed. 
    case '6.4k_dt0.01_idxX1_nEL15'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 6400;         % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Out-of-sample dataset parameters
        Out.dt         = 0.01;         % sampling interval
        Out.Res.beta   = 8/3;          % L63 parameter beta
        Out.Res.rho    = 28;           % L63 parameter rho
        Out.Res.sigma  = 10;           % L63 parameter sigma
        Out.Res.nSProd = 6400;         % number of "production" samples
        Out.Res.nSSpin = 64000;        % spinup samples
        Out.Res.x0     = [ 0.2 1.2 1.25 ]; % initial conditions
        Out.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        Out.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 1;       % observed state vector components 
        In.Src.idxE    = 1 : 16;      % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 500;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nParNN     = 2;          % parallel workers for nearest neighbors
        In.nParE      = 2;          % workers for delay-embedding sums
        In.nN         = 0;          % nearest neighbors for pairwise distances
        In.lDist      = 'l2';       % local distance
        In.tol        = 0;          % 0 distance threshold (for cone kernel)
        In.zeta       = 0;          % cone kernel parameter 
        In.coneAlpha  = 0;          % velocity exponent in cone kernel
        In.nNS        = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = .5;        % diffusion maps normalization 
        In.nPhi        = 1001;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;            % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

        % NLSA parameters, out-of-sample data
        Out.Res.nB    = 1;
        Out.Res.nBRec = 1;




    otherwise
        error( 'Invalid experiment' )
end


%% CHECK IF WE ARE DOING OUT-OF-SAMPLE EXTENSION
ifOse = exist( 'Out', 'var' );

%% CONSTRUCT NLSA MODEL
if ifOse
    Out.Src = In.Src;
    Out.Trg = In.Trg;
    args = { In Out };
else
    args = { In };
end
[ model, In, Out ] = l63NLSAModel( args{ : } );
