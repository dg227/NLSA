function [ model, In, Out ] = demoKoopman_nlsaModel( experiment )
% DEMOKOOPMAN_NLSAMODEL Construct NLSA model for analysis of Lorenz 63 data
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
% to function l63NLSAModel to build the model.
%
% Information about dataset partitioning and parallelization:
%
% The parameter In.nB sets the number of batches to partition the dataset
% in pairwise distance calculations. Increasing this parameter reduces the
% memory footprint, but increases execution time associated with file I/O and 
% nearest-neighbor sorting. As a rule of thumb, values of In.nB between 4
% and 6 are appropriate for L63 datasets with ~50,000 samples on a machine
% with 32GB of memory. Note that enabling parallel for loops (as described 
% below) results in additional memory requirements. 
%
% The parameters In.nParE and In.nParNN set the number of Matlab parallel 
% workers used in parfor loops to compute distances in delay-embedding space 
% and to identify nearest neighbors, respectively. Setting these parameters
% to 0 reverts to serial for loops. 
%
% Modified 2021/09/13

if nargin == 0
    experiment = '16k_dt0.01_nEL800';
end

switch experiment


    % 16000 samples, sampling interval 0.01, 800 delays
    case '16k_dt0.01_nEL800'
        % In-sample dataset parameters
        In.dt         = 0.01;         % sampling interval
        In.Res.beta   = 8/3;          % L63 parameter beta
        In.Res.rho    = 28;           % L63 parameter rho
        In.Res.sigma  = 10;           % L63 parameter sigma
        In.Res.nSProd = 16000;        % number of "production" samples
        In.Res.nSSpin = 64000;        % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;         % relative tolerance for ODE solver 
        In.Res.ifCent = false;        % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 801;     % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;           % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'overlap'; % storage format for delay embedding

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
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = .5;        % diffusion maps normalization 
        In.nPhi        = 401;       % diffusion eigenfunctions to compute
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

        % Koopman generator parameters
        In.koopmanOpType = 'diff';     % Koopman generator type
        In.koopmanFDType  = 'central'; % finite-difference type
        In.koopmanFDOrder = 4;         % finite-difference order
        In.koopmanDt      = 1;         % sampling interval (in months)
        In.koopmanAntisym = true;      % enforce antisymmetrization
        In.koopmanEpsilon = 5E-4;      % regularization parameter
        In.koopmanRegType = 'inv';     % regularization type
        In.idxPhiKoopman  = 1 : 201;   % diffusion eigenfunctions used as basis
        In.nPhiKoopman    = numel( In.idxPhiKoopman ); % Koopman eigenfunctions to compute
        In.nKoopmanPrj    = In.nPhiKoopman; % Koopman eigenfunctions for projection

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
