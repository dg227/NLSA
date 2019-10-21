function [ model, In ] = l96NLSAModel_den( experiment )
%% TORUSNLSAMODEL Build space-time NLSA model with kernel density estimation 
%  for Kuramoto-Sivashinsky system. 
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
% Modidied 2017/07/20
 
if nargin == 0
    experiment = 'test';
end

switch experiment

    case 'test'

        % Default case with variable-bandwidth kernel

        % In-sample dataset parameters
        In.L      = 18;     % system size 
        In.N      = 64;     % number of Fourier modes (= number of spatial gridpoints)
        In.aIC    = 0.6;    % initial condition for Fourier modes
        In.nIC    = 4;      % number of Fourier modes to be initialized to aIC
        In.dt     = 0.25;   % sampling interval 
        In.nSProd = 1E2;    % number of "production" samples
        In.nSSpin = 1E4;    % spinup samples

        % NLSA parameters
        In.nEL          = 15;        % embedding window length 
        In.nXB          = 1;         % samples before main interval (for FD)
        In.nXA          = 1;         % samples after main interval (for FD)
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'overlap'; % storage format for delay embedding
        In.nB           = 1;         % batches to partition the in-sample data
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
        In.nPhi         = 51;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 2 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 2 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 2 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 1;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 100;           % number of exponents for bandwidth tuning

    case 'test_t'

        % Variable-bandwidth kernel with "time change"

        % In-sample dataset parameters
        In.L      = 18;     % system size 
        In.N      = 64;     % number of Fourier modes (= number of spatial gridpoints)
        In.aIC    = 0.6;    % initial condition for Fourier modes
        In.nIC    = 4;      % number of Fourier modes to be initialized to aIC
        In.dt     = 0.25;   % sampling interval 
        In.nSProd = 1E2;    % number of "production" samples
        In.nSSpin = 1E4;    % spinup samples

        % NLSA parameters, kernel density estimation
        In.denType     = 'vb';          % density estimation type
        In.denND       = 1;             % manifold dimension for 
        In.denLDist    = 'l2';          % local distance function
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors 
        In.denZeta     = 0;             % cone kernel parameter
        In.denAlpha    = 0;             % cone kernel velocity exponent
        In.denEpsilonB = 2;             % kernel bandwidth base
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents
        In.denNEpsilon = 100;           % number of exponents for bandwidth tuning

        % NLSA parameters
        In.nEL        = 15;        % embedding window length 
        In.nXB        = 1;         % samples before main interval (for FD)
        In.nXA        = 1;         % samples after main interval (for FD)
        In.fdOrder    = 2;         % finite-difference order 
        In.fdType     = 'central'; % finite-difference type
        In.embFormat  = 'overlap'; % storage format for delay embedding
        In.nB         = 1;         % batches to partition the in-sample data
        In.nBRec      = In.nB;     % batches for reconstructed data
        In.nN         = 1000;      % nearest neighbors for pairwise distances
        In.lDist      = 'cone';    % local distance
        In.tol        = 0;         % 0 distance threshold (for cone kernel)
        In.zeta       = 0;         % cone kernel parameter 
        In.coneAlpha  = -1 / In.denND; % velocity exponent in cone kernel
        In.nNS        = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb';   % diffusion operator type
        In.epsilon    = 1.5;       % kernel bandwidth parameter 
        In.epsilonB   = 2;         % kernel bandwidth base
        In.epsilonE   = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon   = 100;       % number of exponents for bandwidth tuning
        In.alpha      = 1;         % diffusion maps normalization 
        In.nPhi       = 51;        % diffusion eigenfunctions to compute
        In.nPhiPrj    = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec  = 2 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD  = 2 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec   = 2 : 5;     % SVD termporal patterns for reconstruction

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
% Setup nlsaComponent objects 

strSrc = [ 'L'       num2str( In.L, '%1.3g' ) ...
           '_N'      int2str( In.N ) ...
           '_N0'     int2str( In.nIC ) ...
           '_a0'     sprintf( '_%1.3g', In.aIC ) ...
           '_dt'     num2str( In.dt, '%1.3g' ) ...
           '_nS'     int2str( In.nS ) ...
           '_nSSpin' int2str( In.nSSpin ) ];

pathSrc = fullfile( './data', 'raw', strSrc );
if ~isdir( pathSrc )
    mkdir( pathSrc )
end


inPath   = fullfile( './data/raw',  strSrc ); % path for in-sample data
nlsaPath = fullfile( './data/nlsa' );         % path for NLSA code output

tagSrc   = strSrc;                            % tag for in-sample data

% Partition objects 
srcPartition    = nlsaPartition( 'nSample', In.nS );

nR = In.N + 1;
for iR = nR : -1 : 1

    % Filenames
    % dataX.mat must contain an array x of size [ nD nS ], where
    % nD is the dimension and nS the sample number
    srcFilelist = nlsaFilelist( 'file', sprintf( 'dataX_%i.mat', iR ) );

    % nlsaComponent object for in-sample data
    srcComponent( iR ) = nlsaComponent( 'partition',    srcPartition, ...
                                        'dimension',    1, ...
                                        'path',         inPath, ...
                                        'file',         srcFilelist, ...
                                        'componentTag', tagSrc, ...
                                        'realizationTag', sprintf( 'x%i', iR )  );
    embPartition( iR ) = nlsaPartition( 'nSample', In.nSE, 'nBatch',  In.nB  );
end
                             
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
model = nlsaModel_den( 'path',                            nlsaPath, ...
                       'sourceTime',                      In.t, ...
                       'sourceComponent',                 srcComponent, ...
                       'sourceRealizationName',           '', ...
                       'denRealizationName',              '', ...
                       'densityRealizationName',          '', ...
                       'embDensityRealizationName',       '', ...
                       'targetRealizationName',           '', ... 
                       'projectionComponentName',         '', ...
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
                    
