function [ model, In ] = torusNLSAModel_den( experiment )
%% TORUSNLSAMODEL_DEN Build NLSA model with kernel density estimation for a 
%  dynamical system on the 2-torus. 
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
% Modidied 2016/03/22
 
if nargin == 0
    experiment = 'test';
end

switch experiment

    case 'test'
        % In-sample dataset parameters
        In.f       = sqrt( 30 ); % frequency
        In.aPhi    = 1;          % speed variation in phi angle (1=linear flow)
        In.aTheta  = 1;          % speed variation in theta angle 
        In.nST     = 128;         % samples per period
        In.nT      = 32;         % number of periods
        In.nTSpin  = 0;          % spinup periods
        In.nEL     = 0;          % Takens embedding window 
        In.nXB     = 1;          % samples to leave out before main interval
        In.nXA     = 1;          % samples to leave out after main interval
        In.obsMap  = 'r3';       % embedding map 
        In.r1      = .5;         % tube radius along phi angle
        In.r2      = .5;         % tube radius along theta angle
        In.idxX    = [ 1 : 3 ];  % data vector components to observe
        In.p       = 0;          % deformation parameter (in R^3) 
        In.idxP    = 3;          % component to apply deformation
        In.ifCent  = false;      % data centering

        % NLSA parameters
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 3;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 400;       % nearest neighbors for pairwise distances
        In.lDist        = 'at';      % local distance
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
In.nS     = In.nST * In.nT + In.nEL + In.nXB + In.nXA; % total number of samples
In.nSSpin = In.nST * In.nTSpin; % number of spinup samples
In.dt     = 2 * pi / In.nST; % sampling interval
In.t      = linspace( 0, ( In.nS - 1 ) * In.dt, In.nS ); % timestamps
In.nSE    = In.nS - In.idxT1 + 1 - In.nXA; % number of samples after embedding
%==============================================================================
% Setup nlsaComponent objects 

switch In.obsMap
    case 'r3' % embedding in R^3  
        strSrc = [ 'r3' ...
                   '_r1'     num2str( In.r1, '%1.2f' ) ...
                   '_r2'     num2str( In.r2, '%1.2f' ) ...
                   '_f',     num2str( In.f, '%1.2f' ) ...
                   '_aPhi'   num2str( In.aPhi, '%1.2f' ) ...
                   '_aTheta' num2str( In.aTheta, '%1.2f' ) ...
                   '_p'      num2str( In.p ) ...
                   '_idxP'   sprintf( '%i_', In.idxP ) ...
                   'dt'      num2str( In.dt ) ...
                   '_nS'     int2str( In.nS ) ...
                   '_nSSpin' int2str( In.nSSpin ) ...
                   '_idxX'   sprintf( '%i_', In.idxX ) ...  
                   'ifCent'  int2str( In.ifCent ) ];

    case 'r4' % flat embedding in R^4
        strSrc = [ 'r4' ...
                   '_f',     num2str( In.f, '%1.2f' ) ...
                   '_aPhi'   num2str( In.aPhi, '%1.2f' ) ...
                   '_aTheta' num2str( In.aTheta, '%1.2f' ) ...
                   'dt'      num2str( In.dt ) ...
                   '_nS'     int2str( In.nS ) ...
                   '_nSSpin' int2str( In.nSSpin ) ...
                   '_idxX'   sprintf( '%i_', In.idxX ) ...  
                   'ifCent'  int2str( In.ifCent ) ];
end

inPath   = fullfile( './data/raw',  strSrc ); % path for in-sample data
nlsaPath = fullfile( './data/nlsa' );         % path for NLSA code output
tagSrc   = strSrc;                            % tag for in-sample data

nD  = numel( In.idxX );  % data space dimension for in-sample data

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
% Pairwise distance for density estimation
switch In.denLDist
    case 'l2' % L^2 distance
        denLDist = nlsaLocalDistance_l2();

    case 'at' % "autotuning" NLSA kernel
        denLDist = nlsaLocalDistance_at(); 

    case 'cone' % cone kernel
        denLDist = nlsaLocalDistance_cone( 'zeta', In.denZeta, ...
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
        lDist = nlsaLocalDistance_l2();

    case 'at' % "autotuning" NLSA kernel
        lDist = nlsaLocalDistance_at(); 

    case 'cone' % cone kernel
        lDist = nlsaLocalDistance_cone( 'zeta', In.zeta, ...
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
    case 'glb'
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
                    
