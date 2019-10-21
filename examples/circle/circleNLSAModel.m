function [ model, In ] = circleNLSAModel( experiment )
%% CIRCLENLSAMODEL Build NLSA model for a dynamical system on the circle
% 
%  In is a data structure containing the model parameters (named after 
%  "in-sample," as opposed to "out-of-sample" data).
%
%  See script circleData.m for additional details on the dynamical system.
%
%  For additional information on the arguments of nlsaModel( ... ) see 
%
%      ../classes/nlsaModel_base/parseTemplates.m
%      ../classes/nlsaModel/parseTemplates.m
% 
% Modidied 2019/10/19

if nargin == 0
    experiment = 'test';
end
 
switch experiment

    case 'test'
        % In-sample dataset parameters
        In.f       = sqrt( 30 ); % frequency
        In.aTheta  = 1;          % speed variation in theta angle 
        In.nST     = 128;        % samples per period
        In.nT      = 64; 32;         % number of periods
        In.nTSpin  = 0;          % spinup periods
        In.idxE    = [ 1 : 1 ];  % delay embedding indices
        In.nXB     = 1;          % samples to leave out before main interval
        In.nXA     = 1;          % samples to leave out after main interval
        In.obsMap  = 'r2';       % embedding map 
        In.r       = 1;          % tube radius along phi angle
        In.idxX    = [ 1 : 2 ];  % data vector components to observe
        In.ifCent  = false;      % data centering
        
        % NLSA parameters
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'overlap'; % storage format for delay embedding
        In.nB           = 3;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 200;       % nearest neighbors for pairwise distances
        In.lDist        = 'l2';      % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = .99;       % cone kernel parameter 
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

end


%% NUMBER OF SAMPLES, DELAY EMBEDDING
% Determine total number of samples, time origin, and delay-embedding indices

% In-sample data
In.nE     = In.idxE( end ); % max number of delays 
In.idxT1  = In.idxE + In.nXB;    % time origin for delay embedding
In.nS     = In.nST * In.nT + In.nE - 1 + In.nXB + In.nXA; % total number of samples
In.nSSpin = In.nST * In.nTSpin; % number of spinup samples
In.dt     = 2 * pi / In.nST; % sampling interval
In.t      = linspace( 0, ( In.nS - 1 ) * In.dt, In.nS ); % timestamps
In.nSE    = In.nS - In.idxT1 + 1 - In.nXA; % number of samples after embedding


%% IN-SAMPLE DATA COMPONENTS
switch In.obsMap
    case 'r2' % embedding in R^3  
        strSrc = [ 'r2' ...
                   '_r'     num2str( In.r, '%1.2f' ) ...
                   '_f',     num2str( In.f, '%1.2f' ) ...
                   '_aTheta' num2str( In.aTheta, '%1.2f' ) ...
                   '_dt'      num2str( In.dt ) ...
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
                             
%% DELAY-EMBEDDING TEMPLATES
% In-sample data
switch In.embFormat
    case 'evector' % explicit delay-embedded vectors
        embComponent= nlsaEmbeddedComponent_xi_e( 'idxE', In.idxE, ...
                                                  'nXB',  In.nXB, ...
                                                  'nXA', In.nXA, ...
                                                  'fdOrder', In.fdOrder, ...
                                                  'fdType', In.fdType );
        modeStr = 'explicit';
    case 'overlap' % perform delay embedding on the fly
        embComponent= nlsaEmbeddedComponent_xi_o( 'idxE', In.idxE, ...
                                                  'nXB',  In.nXB, ...
                                                  'nXA', In.nXA, ...
                                                  'fdOrder', In.fdOrder, ...
                                                  'fdType', In.fdType );
       modeStr = 'implicit'; 
end

%% PAIRWISE DISTANCES
switch In.lDist
    case 'l2' % L^2 distance
        lDist = nlsaLocalDistance_l2( 'mode', modeStr );

    case 'at' % "autotuning" NLSA kernel
        lDist = nlsaLocalDistance_at( 'mode', modeStr ); 

    case 'cone' % cone kernel
        lDist = nlsaLocalDistance_cone( 'zeta', In.zeta, ...
                                        'tolerance', In.tol, ...
                                        'alpha', In.coneAlpha, ...
                                        'mode', modeStr );
end
dFunc = nlsaLocalDistanceFunction( 'localDistance', lDist );
pDist = nlsaPairwiseDistance( 'distanceFunction', dFunc, ...
                              'nearestNeighbors', In.nN );

%% SYMMETRIZED PAIRWISE DISTANCES
sDist = nlsaSymmetricDistance_gl( 'nearestNeighbors', In.nNS );

%% DIFFUSION OPERATORS
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
end

%% PROJECTIONS AND LINEAR MAP (SVD) OF TARGET DATA
prjComponent = nlsaProjectedComponent( 'nBasisFunction', In.nPhiPrj );
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhiSVD );

%% RECONSTRUCTED COMPONENTS
% Partition
In.nSRec  = In.nSE + In.nE - 1;  % in-sample reconstructed data
recPartition = nlsaPartition( 'nSample', In.nSRec, ... 
                              'nBatch',  In.nBRec );

% Reconstructed data from diffusion eigenfnunctions
recComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxPhiRec );

% Reconstructed data from SVD 
svdRecComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxVTRec );

%% BUILD NLSA MODEL
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
                   'reconstructionTemplate',          recComponent, ...
                   'reconstructionPartition',         recPartition, ...
                   'linearMapTemplate',               linMap, ...
                   'svdReconstructionTemplate',       svdRecComponent );
                    
