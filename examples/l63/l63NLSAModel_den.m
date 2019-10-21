function [ model, In ] = l63NLSAModel_den( experiment )
%% L63NLSAMODEL_DEN Build NLSA model with kernel density estimation for the
%  Lorenz 63 (L63) system. 
% 
%  In is a data structure containing the model parameters (named after 
%  "in-sample," as opposed to "out-of-sample" data).
%
%  This script assumes that solutions of the L63 model have been computed
%  using the script l63Data.m, or any other method that outputs the solutions
%  in the same format as that script.
%
%  For additional information on the arguments of nlsaModel_den( ... ) see 
%
%      ../classes/nlsaModel_base/parseTemplates.m
%      ../classes/nlsaModel/parseTemplates.m
%      ../classes/nlsaModel_den/parseTemplates.m
%
% Modidied 2018/05/03

if nargin == 0
    experiment = '64k';
end

switch experiment

    case 'test'
        % In-sample dataset parameters
        In.beta   = 8/3;   % L63 parameter beta
        In.rho    = 28;    % L63 parameter rho
        In.sigma  = 10;    % L63 parameter sigma
        In.nSProd = 64; % number of "production samples
        In.nSSpin = 64; % spinup samples
        In.nEL    = 0;     % embedding window length (additional samples)
        In.nXB    = 0;     % additional samples before production interval (for FD)
        In.nXA    = 0;     % additional samples after production interval (for FD)
        In.dt     = 0.01;  % sampling interval
        In.x0     = [ 0 1 1.05 ]; % initial conditions
        In.relTol = 1E-8;  % relative tolerance for ODE solver 
        In.ifCent = false; % data centering
        In. nS     =   In.nSProd + In.nEL + In.nXB + In.nXA; % sample number

        % NLSA parameters
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'overlap'; % storage format for delay embedding
        In.nB           = 1;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 50;      % nearest neighbors for pairwise distances
        In.lDist        = 'l2';      % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;         % cone kernel parameter 
        In.coneAlpha    = 0;         % velocity exponent in cone kernel
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb_svd';   % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 200;       % number of exponents for bandwidth tuning
        In.alpha        = 1;         % diffusion maps normalization 
        In.nPhi         = 51;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 2;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;           % number of exponents for bandwidth tuning


    % 64000 samples, standard L63 parameters
    case '64k'
        % In-sample dataset parameters
        In.beta   = 8/3;   % L63 parameter beta
        In.rho    = 28;    % L63 parameter rho
        In.sigma  = 10;    % L63 parameter sigma
        In.nSProd = 64000; % number of "production samples
        In.nSSpin = 64000; % spinup samples
        In.nEL    = 0;     % embedding window length (additional samples)
        In.nXB    = 0;     % additional samples before production interval (for FD)
        In.nXA    = 0;     % additional samples after production interval (for FD)
        In.dt     = 0.01;  % sampling interval
        In.x0     = [ 0 1 1.05 ]; % initial conditions
        In.relTol = 1E-8;  % relative tolerance for ODE solver 
        In.ifCent = false; % data centering
        In. nS     =   In.nSProd + In.nEL + In.nXB + In.nXA; % sample number

        % NLSA parameters
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'overlap'; % storage format for delay embedding
        In.nB           = 8;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 5000;      % nearest neighbors for pairwise distances
        In.lDist        = 'l2';      % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;         % cone kernel parameter 
        In.coneAlpha    = 0;         % velocity exponent in cone kernel
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';   % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 200;       % number of exponents for bandwidth tuning
        In.alpha        = 1;         % diffusion maps normalization 
        In.nPhi         = 51;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 2;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;           % number of exponents for bandwidth tuning

    % 64000 samples, standard L63 parameters, time change
    case '64k_cone'
        % In-sample dataset parameters
        In.beta   = 8/3;   % L63 parameter beta
        In.rho    = 28;    % L63 parameter rho
        In.sigma  = 10;    % L63 parameter sigma
        In.nSProd = 64000; % number of "production samples
        In.nSSpin = 64000; % spinup samples
        In.nEL    = 0;     % embedding window length (additional samples)
        In.nXB    = 1;     % additional samples before production interval (for FD)
        In.nXA    = 1;     % additional samples after production interval (for FD)
        In.dt     = 0.01;  % sampling interval
        In.x0     = [ 0 1 1.05 ]; % initial conditions
        In.relTol = 1E-8;  % relative tolerance for ODE solver 
        In.ifCent = false; % data centering
        In. nS     =   In.nSProd + In.nEL + In.nXB + In.nXA; % sample number

        % NLSA parameters
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 8;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 5000;      % nearest neighbors for pairwise distances
        In.lDist        = 'cone';    % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = .99;       % cone kernel parameter 
        In.coneAlpha    = 0;         % cone kernel velocity exponent (time change)
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';   % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 100;       % number of exponents for bandwidth tuning
        In.alpha        = 1;         % diffusion maps normalization 
        In.nPhi         = 501;       % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 2;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;           % number of exponents for bandwidth tuning

% 64000 samples, standard L63 parameters, time change
    case '64k_cone0'
        % In-sample dataset parameters
        In.beta   = 8/3;   % L63 parameter beta
        In.rho    = 28;    % L63 parameter rho
        In.sigma  = 10;    % L63 parameter sigma
        In.nSProd = 64000; % number of "production samples
        In.nSSpin = 64000; % spinup samples
        In.nEL    = 0;     % embedding window length (additional samples)
        In.nXB    = 1;     % additional samples before production interval (for FD)
        In.nXA    = 1;     % additional samples after production interval (for FD)
        In.dt     = 0.01;  % sampling interval
        In.x0     = [ 0 1 1.05 ]; % initial conditions
        In.relTol = 1E-8;  % relative tolerance for ODE solver 
        In.ifCent = false; % data centering
        In. nS     =   In.nSProd + In.nEL + In.nXB + In.nXA; % sample number

        % NLSA parameters
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 8;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 5000;      % nearest neighbors for pairwise distances
        In.lDist        = 'cone';    % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;       % cone kernel parameter 
        In.coneAlpha    = 0;         % cone kernel velocity exponent (time change)
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';   % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 100;       % number of exponents for bandwidth tuning
        In.alpha        = 1;         % diffusion maps normalization 
        In.nPhi         = 501;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 2;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;           % number of exponents for bandwidth tuning


    case '64k_cone0_q300'
        % In-sample dataset parameters
        In.beta   = 8/3;   % L63 parameter beta
        In.rho    = 28;    % L63 parameter rho
        In.sigma  = 10;    % L63 parameter sigma
        In.nSProd = 64000; % number of "production samples
        In.nSSpin = 64000; % spinup samples
        In.nEL    = 300;     % embedding window length (additional samples)
        In.nXB    = 1;     % additional samples before production interval (for FD)
        In.nXA    = 1;     % additional samples after production interval (for FD)
        In.dt     = 0.01;  % sampling interval
        In.x0     = [ 0 1 1.05 ]; % initial conditions
        In.relTol = 1E-8;  % relative tolerance for ODE solver 
        In.ifCent = false; % data centering
        In. nS     =   In.nSProd + In.nEL + In.nXB + In.nXA; % sample number

        % NLSA parameters
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'overlap'; % storage format for delay embedding
        In.nB           = 8;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 5000;      % nearest neighbors for pairwise distances
        In.lDist        = 'cone';    % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;       % cone kernel parameter 
        In.coneAlpha    = 0;         % cone kernel velocity exponent (time change)
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';   % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 100;       % number of exponents for bandwidth tuning
        In.alpha        = 1;         % diffusion maps normalization 
        In.nPhi         = 501;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 2;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;           % number of exponents for bandwidth tuning



    % 64000 samples, standard L63 parameters, time change
    case '64k_t'
        % In-sample dataset parameters
        In.beta   = 8/3;   % L63 parameter beta
        In.rho    = 28;    % L63 parameter rho
        In.sigma  = 10;    % L63 parameter sigma
        In.nSProd = 64000; % number of "production samples
        In.nSSpin = 64000; % spinup samples
        In.nEL    = 0;     % embedding window length (additional samples)
        In.nXB    = 2;     % additional samples before production interval (for FD)
        In.nXA    = 2;     % additional samples after production interval (for FD)
        In.dt     = 0.01;  % sampling interval
        In.x0     = [ 0 1 1.05 ]; % initial conditions
        In.relTol = 1E-8;  % relative tolerance for ODE solver 
        In.ifCent = false; % data centering
        In. nS     =   In.nSProd + In.nEL + In.nXB + In.nXA; % sample number

        % NLSA parameters
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 8;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 5000;      % nearest neighbors for pairwise distances
        In.lDist        = 'cone';    % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;         % cone kernel parameter 
        In.coneAlpha    = -1 / In.denND; % cone kernel velocity exponent (time change)
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';   % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 100;       % number of exponents for bandwidth tuning
        In.alpha        = 1;         % diffusion maps normalization 
        In.nPhi         = 51;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 2;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;           % number of exponents for bandwidth tuning
end

%% NLSA MODEL
%==============================================================================
% Determine total number of samples, time origin, and delay-embedding indices

% In-sample data
In.idxE   = [ 1 : 1 : In.nEL + 1 ]; % delay embedding indices
In.idxT1  = In.nEL + 1 + In.nXB;    % time origin for delay embedding
In.t      = linspace( 0, ( In.nS - 1 ) * In.dt, In.nS ); % timestamps
In.nSE    = In.nS - In.idxT1 + 1 - In.nXA; % number of samples after embedding

%==============================================================================
% Setup nlsaComponent objects 
strSrc = [ 'beta'    num2str( In.beta, '%1.3g' ) ...
           '_rho'    num2str( In.rho, '%1.3g' ) ...
           '_sigma'  num2str( In.sigma, '%1.3g' ) ...
           '_dt'     num2str( In.dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', In.x0 ) ...
           '_nS'     int2str( In.nS ) ...
           '_nSSpin'  int2str( In.nSSpin ) ...
           '_relTol' num2str( In.relTol, '%1.3g' ) ...
           '_ifCent'  int2str( In.ifCent ) ];

inPath   = fullfile( './data/raw',  strSrc );
nlsaPath = fullfile( './data/nlsa' );
tagSrc   = strSrc;

nD =  3; % data space dimension for in-sample data

% Partition objects 
srcPartition    = nlsaPartition( 'nSample', In.nS );
embPartition    = nlsaPartition( 'nSample', In.nSE, 'nBatch',  In.nB  );

% Filenames
% dataX.mat must contain an array x of size [ nD nS ], where
% nD is the dimension and nS the sample number
srcFilelist = nlsaFilelist( 'file', 'dataX.mat' );

% nlsaComponent object for in-sample data
srcComponent = nlsaComponent( 'partition',    srcPartition, ...
                              'dimension',    nD, ...
                              'path',         inPath, ...
                              'file',         srcFilelist, ...
                              'componentTag', tagSrc  );

%==============================================================================
% Setup delay-embedding templates 

% In-sample data
if In.nXB == 0 && In.nXA == 0
    embComponent= nlsaEmbeddedComponent_o( 'idxE', In.idxE, ...
                                           'nXB',  In.nXB, ...
                                           'nXA', In.nXA );

else
    embComponent= nlsaEmbeddedComponent_xi_o( 'idxE', In.idxE, ...
                                              'nXB',  In.nXB, ...
                                              'nXA', In.nXA, ...
                                              'fdOrder', In.fdOrder, ...
                                              'fdType', In.fdType );
end

%==============================================================================
% Pairwise distance for density estimation
switch In.denLDist
    case 'l2' % L^2 distance
        denLDist = nlsaLocalDistance_l2( 'mode', 'implicit' );

    case 'at' % "autotuning" NLSA kernel
        denLDist = nlsaLocalDistance_at( 'mode', 'implicit' );

    case 'cone' % cone kernel
        denLDist = nlsaLocalDistance_cone( 'mode', 'implicit', ...
                                           'zeta', In.denZeta, ...
                                           'tolerance', In.tol, ...
                                           'alpha', In.denConeAlpha );
end

denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

denPDist = nlsaPairwiseDistance( 'nearestNeighbors', In.nN, ...
                                 'distanceFunction', denDFunc );

%==============================================================================
% Kernel density estimation
switch In.denType
    case 'fb' % fixed bandwidth
        den = nlsaKernelDensity_fb( ...
                 'dimension',              In.denND, ...
                 'bandwidthBase',          In.denEpsilonB, ...
                 'bandwidthExponentLimit', In.denEpsilonE, ...
                 'nBandwidth',             In.denNEpsilon );

    case 'vb' % variable bandwidth 
        den = nlsaKernelDensity_vb( ...
                 'dimension',              In.denND, ...
                 'kNN',                    In.denNN, ...
                 'bandwidthBase',          In.denEpsilonB, ...
                 'bandwidthExponentLimit', In.denEpsilonE, ...
                 'nBandwidth',             In.denNEpsilon );
end

%==============================================================================
% Pairwise distance 
switch In.lDist
    case 'l2' % L^2 distance
        lDist = nlsaLocalDistance_l2( 'mode', 'implicit' );

    case 'at' % "autotuning" NLSA kernel
        lDist = nlsaLocalDistance_at( 'mode', 'implicit' );

    case 'cone' % cone kernel
        lDist = nlsaLocalDistance_cone( 'mode', 'implicit', ...
                                        'zeta', In.zeta, ...
                                        'tolerance', In.tol, ...
                                        'alpha', In.coneAlpha );
end
lScl  = nlsaLocalScaling_pwr( 'pwr', 1 / In.denND );
dFunc = nlsaLocalDistanceFunction_scl( 'localDistance', lDist, ...
                                       'localScaling', lScl );
pDist = nlsaPairwiseDistance( 'distanceFunction', dFunc, ...
                              'nearestNeighbors', In.nN );



%==============================================================================
% Symmetrized pairwise distances
sDist = nlsaSymmetricDistance_gl( 'nearestNeighbors', In.nNS );

%==============================================================================
% Diffusion operators 

switch In.diffOpType
    % global storage format, fixed bandwidth
    case 'gl'
        diffOp = nlsaDiffusionOperator_gl( 'alpha',          In.alpha, ...
                                           'epsilon',        In.epsilon, ...
                                           'nEigenfunction', In.nPhi );

    % global storage format, multiple bandwidth (automatic bandwidth selection)
    case 'gl_mb'
        diffOp = nlsaDiffusionOperator_gl_mb( ...
                     'alpha',                  In.alpha, ...
                     'epsilon',                In.epsilon, ...
                     'nEigenfunction',         In.nPhi, ...
                     'bandwidthBase',          In.epsilonB, ...
                     'bandwidthExponentLimit', In.epsilonE, ...
                     'nBandwidth',             In.nEpsilon );
    case 'gl_mb_svd'
        diffOp = nlsaDiffusionOperator_gl_mb_svd( ...
                     'alpha',                  In.alpha, ...
                     'epsilon',                In.epsilon, ...
                     'nEigenfunction',         In.nPhi, ...
                     'bandwidthBase',          In.epsilonB, ...
                     'bandwidthExponentLimit', In.epsilonE, ...
                     'nBandwidth',             In.nEpsilon );

end

%==============================================================================
% Projections and linear map for SVD of the target data 
prjComponent = nlsaProjectedComponent( 'nBasisFunction', In.nPhiPrj );
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhiSVD );

%==============================================================================
% Reconstructed components

% Partition
In.nSRec  = In.nSE + In.nEL;  % in-sample reconstructed data
recPartition = nlsaPartition( 'nSample', In.nSRec, ... 
                              'nBatch',  In.nBRec );

% Reconstructed data from diffusion eigenfnunctions
recComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxPhiRec );

% Reconstructed data from SVD 
svdRecComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxVTRec );

%==============================================================================
% Build NLSA model    
model = nlsaModel_den( 'path',                            nlsaPath, ...
                       'sourceTime',                      In.t, ...
                       'sourceComponent',                 srcComponent, ...
                       'embeddingOrigin',                 In.idxT1, ...
                       'embeddingTemplate',               embComponent, ...
                       'embeddingPartition',              embPartition, ...
                       'denPairwiseDistanceTemplate',     denPDist, ...
                       'kernelDensityTemplate',           den, ...
                       'pairwiseDistanceTemplate',        pDist, ...
                       'symmetricDistanceTemplate',       sDist, ...
                       'diffusionOperatorTemplate',       diffOp, ...
                       'projectionTemplate',              prjComponent, ...
                       'reconstructionPartition',         recPartition, ...
                       'reconstructionTemplate',          recComponent, ...
                       'linearMapTemplate',               linMap, ...
                       'svdReconstructionTemplate',       svdRecComponent );



