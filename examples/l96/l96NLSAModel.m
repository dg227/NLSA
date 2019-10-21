function [ model, In ] = l96NLSAModel( experiment )
%% TORUSNLSAMODEL Build NLSA model for the Lorenz 96 system. 
% 
%  In is a data structure containing the model parameters (named after 
%  "in-sample," as opposed to "out-of-sample" data).
%
%  See script torusData.m for additional details on the dynamical system.
%
%  For additional information on the arguments of nlsaModel( ... ) see 
%
%      ../classes/nlsaModel_base/parseTemplates.m
%      ../classes/nlsaModel/parseTemplates.m
%      ../classes/nlsaModel_den/parseTemplates.m
%
% Modidied 2016/02/05
 
if nargin == 0
    experiment = 'test';
end

switch experiment

    case 'test'
        % In-sample dataset parameters
        In.n       = 41;            % number of nodes
        In.F       = 4;             % forcing parameter
        In.dt      = 0.01;          % sampling interval 
        In.nSProd  = 1024;          % number of "production" samples
        In.nSSpin  = 16000;         % spinup samples
        In.x0      = 1;             % initial conditions
        In.relTol  = 1E-8;          % integrator tolerance
        In.ifCent  = false;         % data centering

        % NLSA parameters
        In.nEL          = 0;         % embedding window length 
        In.nXB          = 1;         % samples before main interval (for FD)
        In.nXA          = 1;         % samples after main interval (for FD)
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 4;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 1000;      % nearest neighbors for pairwise distances
        In.lDist        = 'l2';      % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;         % cone kernel parameter 
        In.coneAlpha    = 0;         % velocity exponent in cone kernel
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';   % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 100;       % number of exponents for bandwidth tuning
        In.alpha        = 1;         % diffusion maps normalization 
        In.nPhi         = 101;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 2 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 2 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 2 : 5;     % SVD termporal patterns for reconstruction

    case 'F4'
        % In-sample dataset parameters
        In.n       = 41;            % number of nodes
        In.F       = 4;             % forcing parameter
        In.dt      = 0.01;          % sampling interval 
        In.nSProd  = 64000;          % number of "production" samples
        In.nSSpin  = 64000;         % spinup samples
        In.x0      = 1;             % initial conditions
        In.relTol  = 1E-8;          % integrator tolerance
        In.ifCent  = false;         % data centering

        % NLSA parameters
        In.nEL          = 0;         % embedding window length 
        In.nXB          = 1;         % samples before main interval (for FD)
        In.nXA          = 1;         % samples after main interval (for FD)
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 8;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 5000;       % nearest neighbors for pairwise distances
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
        In.nPhi         = 301;       % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 2 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 2 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 2 : 5;     % SVD termporal patterns for reconstruction


    case 'F5'
        % In-sample dataset parameters
        In.n       = 41;            % number of nodes
        In.F       = 5;             % forcing parameter
        In.dt      = 0.01;          % sampling interval 
        In.nSProd  = 128000;        % number of "production" samples
        In.nSSpin  = 128000;        % spinup samples
        In.x0      = 1;             % initial conditions
        In.relTol  = 1E-8;          % integrator tolerance
        In.ifCent  = false;         % data centering

        % NLSA parameters
        In.nEL          = 0;         % embedding window length 
        In.nXB          = 1;         % samples before main interval (for FD)
        In.nXA          = 1;         % samples after main interval (for FD)
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 8;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 10000;     % nearest neighbors for pairwise distances
        In.lDist        = 'l2';      % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;         % cone kernel parameter 
        In.coneAlpha    = 0;         % velocity exponent in cone kernel
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';   % diffusion operator type
        In.epsilon      = 2;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 200;       % number of exponents for bandwidth tuning
        In.alpha        = .5;        % diffusion maps normalization 
        In.nPhi         = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 2 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 2 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 2 : 5;     % SVD termporal patterns for reconstruction


    case 'F8'
        % In-sample dataset parameters
        In.n       = 41;            % number of nodes
        In.F       = 8;             % forcing parameter
        In.dt      = 0.01;          % sampling interval 
        In.nSProd  = 8192;          % number of "production" samples
        In.nSSpin  = 16000;         % spinup samples
        In.x0      = 1;             % initial conditions
        In.relTol  = 1E-8;          % integrator tolerance
        In.ifCent  = false;         % data centering

        % NLSA parameters
        In.nEL          = 0;         % embedding window length 
        In.nXB          = 1;         % samples before main interval (for FD)
        In.nXA          = 1;         % samples after main interval (for FD)
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 4;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 1000;       % nearest neighbors for pairwise distances
        In.lDist        = 'l2';      % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;         % cone kernel parameter 
        In.coneAlpha    = 0;         % velocity exponent in cone kernel
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';   % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 100;       % number of exponents for bandwidth tuning
        In.alpha        = 1;         % diffusion maps normalization 
        In.nPhi         = 51;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 2 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 2 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 2 : 5;     % SVD termporal patterns for reconstruction


end


%% NLSA MODEL

%==============================================================================
% Determine total number of samples, time origin, and delay-embedding indices

% In-sample data
In.idxE   = [ 1 : 1 : In.nEL + 1 ]; % delay embedding indices
In.idxT1  = In.nEL + 1 + In.nXB;    % time origin for delay embedding
In.nS     = In.nSProd + In.nEL + In.nXB + In.nXA; % total number of samples 
In.t      = linspace( 0, ( In.nS - 1 ) * In.dt, In.nS ); % timestamps
In.nSE    = In.nS - In.idxT1 + 1 - In.nXA; % number of samples after embedding


%==============================================================================
% Fill initial conditions
In.x0 = [ In.x0 zeros( 1, In.n - 1 ) ];

%==============================================================================
% Setup nlsaComponent objects 

strSrc = [ 'F'       num2str( In.F, '%1.3g' ) ...
           '_n'      int2str( In.n ) ...
           '_dt'     num2str( In.dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', In.x0( 1  ) ) ...
           '_nS'     int2str( In.nS ) ...
           '_nSSpin' int2str( In.nSSpin ) ...
           '_relTol' num2str( In.relTol, '%1.3g' ) ...
           '_ifCent' int2str( In.ifCent ) ];

inPath   = fullfile( './data/raw',  strSrc ); % path for in-sample data
nlsaPath = fullfile( './data/nlsa' );         % path for NLSA code output
tagSrc   = strSrc;                            % tag for in-sample data

nD  = In.n;  % data space dimension for in-sample data

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
switch In.embFormat
    case 'evector' % explicit delay-embedded vectors
        embComponent= nlsaEmbeddedComponent_xi_e( 'idxE', In.idxE, ...
                                                  'nXB',  In.nXB, ...
                                                  'nXA', In.nXA, ...
                                                  'fdOrder', In.fdOrder, ...
                                                  'fdType', In.fdType );
    case 'overlap' % perform delay embedding on the fly
        embComponent= nlsaEmbeddedComponent_xi_o( 'idxE', In.idxE, ...
                                                  'nXB',  In.nXB, ...
                                                  'nXA', In.nXA, ...
                                                  'fdOrder', In.fdOrder, ...
                                                  'fdType', In.fdType );

end

%==============================================================================
% Pairwise distance 
switch In.lDist
    case 'l2' % L^2 distance
        lDist = nlsaLocalDistance_l2();

    case 'at' % "autotuning" NLSA kernel
        lDist = nlsaLocalDistance_at(); 

    case 'cone' % cone kernel
        lDist = nlsaLocalDistance_cone( 'zeta', In.zeta, ...
                                        'tolerance', In.tol, ...
                                        'alpha', In.coneAlpha );
end
dFunc = nlsaLocalDistanceFunction( 'localDistance', lDist );
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
        diffOp = nlsaDiffusionOperator_gl_fb( 'alpha',          In.alpha, ...
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
model = nlsaModel( 'path',                            nlsaPath, ...
                   'sourceTime',                      In.t, ...
                   'sourceComponent',                 srcComponent, ...
                   'embeddingOrigin',                 In.idxT1, ...
                   'embeddingTemplate',               embComponent, ...
                   'embeddingPartition',              embPartition, ...
                   'pairwiseDistanceTemplate',        pDist, ...
                   'symmetricDistanceTemplate',       sDist, ...
                   'diffusionOperatorTemplate',       diffOp, ...
                   'projectionTemplate',              prjComponent, ...
                   'reconstructionPartition',         recPartition, ...
                   'reconstructionTemplate',          recComponent, ...
                   'linearMapTemplate',               linMap, ...
                   'svdReconstructionTemplate',       svdRecComponent );
                    
