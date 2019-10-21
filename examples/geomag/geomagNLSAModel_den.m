function [ model, In ] = geomagNLSAModel_den( experiment )

if nargin == 0
    experiment = 'oneyarindex2000';
end
 
switch experiment

    case 'AL_SW_2000'
        % In-sample dataset parameters
        In.fld     = 'AL_SW';
        In.tLim    = [ 1 10000 ];
        In.nSkip   = 3;
        In.nShift  = 0; 
        In.nD      = 2;  % data space dimension for in-sample data

        % NLSA parameters, in-sample data
        In.nEL         = 10;        % Takens embedding window length
        In.nXB         = 1;         % samples to leave out before main interval
        In.nXA         = 1;         % samples to leave out after main interval
        In.fdOrder     = 2;         % finite-difference order 
        In.fdType      = 'central'; % finite-difference type
        In.embFormat   = 'evector'; % storage format for delay embedding
        In.nB          = 8;         % batches to partition the in-sample data
        In.nBRec       = In.nB;     % batches for reconstructed data
        In.nN          = 200;       % nearest neighbors for pairwise distances
        In.denType     = 'vb';      % density estimation type
        In.denND       = 3;         % manifold dimension for KDE
        In.denLDist    = 'l2';      % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;         % nearest neighbors for KDE
        In.denZeta     = 0;         % cone kernel parameter (for KDE)
        In.denAlpha    = 0;         % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;         % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ]; % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;       % number of exponents for bandwidth tuning
        In.lDist       = 'l2';      % local distance function
        In.tol         = 0;         % 0 distance threshold (for cone kernel)
        In.zeta        = 0;         % cone kernel parameter 
        In.coneAlpha   = 0; - 1 / In.denND; % cone kernel velocity exponent 
% 0 -> no time change
% - 1 / In.denND -> time change
        In.nNS         = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType  = 'gl_mb';   % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -40 40 ]; % kernel bandwidth exponents 
        In.nEpsilon    = 100;       % number of exponents for bandwidth tuning
        In.alpha       = 1;        % diffusion maps normalization 
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 15;    % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

    case 'AL_2000'
        % In-sample dataset parameters
        In.fld     = 'AL';
        In.tLim    = [ 1 1000 ];
        In.nSkip   = 3;
        In.nShift  = 0;
        In.nD      = 1;  % data space dimension for in-sample data

        % NLSA parameters, in-sample data
        In.nEL         = 10;        % Takens embedding window length
        In.nXB         = 1;         % samples to leave out before main interval
        In.nXA         = 1;         % samples to leave out after main interval
        In.fdOrder     = 2;         % finite-difference order 
        In.fdType      = 'central'; % finite-difference type
        In.embFormat   = 'evector'; % storage format for delay embedding
        In.nB          = 8;         % batches to partition the in-sample data
        In.nBRec       = In.nB;     % batches for reconstructed data
        In.nN          = 500;       % nearest neighbors for pairwise distances
        In.denType     = 'vb';      % density estimation type
        In.denND       = 3;         % manifold dimension for KDE
        In.denLDist    = 'l2';      % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;         % nearest neighbors for KDE
        In.denZeta     = 0;         % cone kernel parameter (for KDE)
        In.denAlpha    = 0;         % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;         % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ]; % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;       % number of exponents for bandwidth tuning
        In.lDist       = 'l2';      % local distance function
        In.tol         = 0;         % 0 distance threshold (for cone kernel)
        In.zeta        = 0;         % cone kernel parameter 
        In.coneAlpha   = 0; - 1 / In.denND; % cone kernel velocity exponent 
% 0 -> no time change
% - 1 / In.denND -> time change
        In.nNS         = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType  = 'gl_mb';   % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -40 40 ]; % kernel bandwidth exponents 
        In.nEpsilon    = 100;       % number of exponents for bandwidth tuning
        In.alpha       = 1;        % diffusion maps normalization 
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 15;    % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction
end


%% NLSA MODEL

%==============================================================================
% Determine total number of samples, time origin, and delay-embedding indices

% In-sample data
In.idxE   = [ 1 : 1 : In.nEL + 1 ]; % delay embedding indices
In.idxT1  = In.nEL + 1 + In.nXB;    % time origin for delay embedding
In.nS     = numel( In.tLim( 1 ) : In.nSkip : In.tLim( 2 ) ); % total number of samples
In.t      = linspace( 0, ( In.nS - 1 ) * In.nSkip, In.nS ); % timestamps
In.nSE    = In.nS - In.idxT1 + 1 - In.nXA; % number of samples after embedding
%==============================================================================
% Setup nlsaComponent objects 

strSrc = sprintf( '%i-%i_nSkip%i_nShift%i', In.tLim( 1 ), In.tLim( 2 ), In.nSkip, In.nShift );
inPath   = fullfile( './data/raw', In.fld ); % path for in-sample data
nlsaPath = fullfile( './data/nlsa' );         % path for NLSA code output
tagSrc   = strSrc;                            % tag for in-sample data


% Partition objects 
srcPartition    = nlsaPartition( 'nSample', In.nS );
embPartition    = nlsaPartition( 'nSample', In.nSE, 'nBatch',  In.nB  );

% Filenames
% dataX.mat must contain an array x of size [ nD nS ], where
% nD is the dimension and nS the sample number
srcFilelist = nlsaFilelist( 'file', [ 'dataX_' strSrc '.mat' ] );

% nlsaComponent object for in-sample data
srcComponent = nlsaComponent( 'partition',    srcPartition, ...
                              'dimension',    In.nD, ...
                              'path',         inPath, ...
                              'file',         srcFilelist, ...
                              'componentTag', In.fld, ...
                              'realizationTag', strSrc  );
                             
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
    case 'gl_fb'
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
In.nSRec  = In.nSE + In.nEL;  % in-sample reconstructed data
recPartition = nlsaPartition( 'nSample', In.nSRec, ... 
                              'nBatch',  In.nBRec );

% Reconstructed data from diffusion eigenfnunctions
recComponent = nlsaComponent_rec_phi( 'partition', recPartition, ...
                                      'basisFunctionIdx', In.idxPhiRec );

% Reconstructed data from SVD 
svdRecComponent = nlsaComponent_rec_phi( 'partition', recPartition, ...
                                         'basisFunctionIdx', In.idxVTRec );

%==============================================================================
% Build NLSA model    
model = nlsaModel_den( 'path',                            nlsaPath, ...
                       'srcTime',                         In.t, ...
                       'sourceComponent',                 srcComponent, ...
                       'embeddingOrigin',                 In.idxT1, ...
                       'embeddingTemplate',               embComponent, ...
                       'partitionTemplate',               embPartition, ...
                       'denPairwiseDistanceTemplate',     denPDist, ...
                       'kernelDensityTemplate',           den, ...
                       'pairwiseDistanceTemplate',        pDist, ...
                       'symmetricDistanceTemplate',       sDist, ...
                       'diffusionOperatorTemplate',       diffOp, ...
                       'projectionTemplate',              prjComponent, ...
                       'reconstructionTemplate',          recComponent, ...
                       'linearMapTemplate',               linMap, ...
                       'svdReconstructionTemplate',       svdRecComponent );
                    
