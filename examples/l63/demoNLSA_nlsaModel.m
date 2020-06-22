function [ model, In, Out ] = demoNLSA_nlsaModel( experiment )
% DEMONLSA_NLSAMODEL Construct NLSA model for analysis of Lorenz 63 data
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
% This function creates the data structures In and Out, which are then passed 
% to function climateNLSAModel_base to build the model.
%
if nargin == 0
    experiment = '6.4k_dt0.01_nEL0';
end

switch experiment

    % 6400 samples, sampling interval 0.01, no delay embedding 
    case '6.4k_dt0.01_nEL0'
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

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 1;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;         % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
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
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % 6400 samples, sampling interval 0.01, 100 delays
    case '6.4k_dt0.01_nEL80'
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

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 81;     % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 4;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
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
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning


    % 6400 samples, sampling interval 0.01, 100 delays
    case '6.4k_dt0.01_nEL100'
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

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 101;     % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 4;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
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
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning


    % 6400 samples, sampling interval 0.01, 150 delays
    case '6.4k_dt0.01_nEL150'
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

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 151;     % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 4;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
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
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning


    % 6400 samples, sampling interval 0.01, 200 delays
    case '6.4k_dt0.01_nEL200'
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

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 201;     % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 4;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
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
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning


    % 6400 samples, sampling interval 0.01, 300 delays
    case '6.4k_dt0.01_nEL300'
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

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 301;     % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 4;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
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
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % 6400 samples, sampling interval 0.01, 300 delays
    case '6.4k_dt0.01_nEL400'
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

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 401;     % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 4;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
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
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 8;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning


    % 64000 samples, sampling interval 0.01, no delay embedding 
    case '64k_dt0.01_nEL0'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 64000;        % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 1;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;         % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nN         = 7000;       % nearest neighbors for pairwise distances
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
        In.nPhi        = 101;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % 64000 samples, sampling interval 0.01, 400 delays 
    case '64k_dt0.01_nEL400'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 64000;        % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 401;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;         % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;   % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;         % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 2;          % batches to partition the in-sample data
        In.Res.nBRec  = 2;          % batches for reconstructed data
        In.nN         = 7000;       % nearest neighbors for pairwise distances
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
        In.nPhi        = 101;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % 64000 samples, sampling interval 0.01, 800 delays 
    case '64k_dt0.01_nEL800'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 64000;        % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 801;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;         % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;   % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;         % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 2;          % batches to partition the in-sample data
        In.Res.nBRec  = 2;          % batches for reconstructed data
        In.nN         = 7000;       % nearest neighbors for pairwise distances
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
        In.nPhi        = 101;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % 64000 samples, sampling interval 0.01, 1600 delays 
    case '64k_dt0.01_nEL1600'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 64000;        % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 1601;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;         % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;   % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;         % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nN         = 7000;       % nearest neighbors for pairwise distances
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
        In.nPhi        = 101;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % 64000 samples, sampling interval 0.01, 3200 delays 
    case '64k_dt0.01_nEL3200'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 64000;        % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 3201;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;         % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'evector'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;   % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;         % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nN         = 7000;       % nearest neighbors for pairwise distances
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
        In.nPhi        = 101;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 2;             % manifold dimension for 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;             % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning


    otherwise
        error( 'Invalid experiment' )
end


%% CHECK IF WE ARE DOING OUT-OF-SAMPLE EXTENSION
ifOse = exist( 'Out', 'var' );

%% CONSTRUCT NLSA MODEL
if ifOse
    args = { In Out };
else
    args = { In };
end
[ model, In, Out ] = l63NLSAModel( args{ : } );
