function [ model, In, Out ] = torusNLSAModel_ose( experiment )
%% TORUSNLSAMODEL_OSE Build NLSA model for a dynamical system on the 
%  2-torus with out-of-sample extension (OSE)
% 
%  In and Out are data structures containing the model parameters
%
%  See script torusData.m for additional details on the dynamical system
%
%  For additional information on the arguments of nlsaModel_ose( ... ) 
%  see 
%
%      nlsa/classes/nlsaModel_base/parseTemplates.m
%      nlsa/classes/nlsaModel/parseTemplates.m
%      nlsa/classes/nlsaModel_ose/parseTemplates.m
% 
% Modidied 2016/04/12
 
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
        
        % Out-of-sample dataset parameters
        Out.f       = sqrt( 30 ); % frequency
        Out.aPhi    = 1;          % speed variation in phi angle (1=linear flow)
        Out.aTheta  = 1;          % speed variation in theta angle 
        Out.nT      = 16;         % number of periods
        Out.nTSpin  = 1024;       % spinup periods
        Out.nXB     = 1;          % samples to leave out before main interval
        Out.nXA     = 1;          % samples to leave out after main interval
        Out.r1      = .5;         % tube radius along phi angle
        Out.r2      = .5;         % tube radius along theta angle
        Out.p       = 0;          % deformation parameter (in R^3)
        Out.idxP    = 3;          % component to apply deformation
        Out.ifCent  = false;      % data centering

        % NLSA parameters, in-sample data
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'evector'; % storage format for delay embedding
        In.nB           = 3;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 200;       % nearest neighbors for pairwise distances
        In.lDist        = 'at';      % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;         % cone kernel parameter 
        In.coneAlpha    = 0;         % velocity exponent in cone kernel
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl';      % diffusion operator type
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

        % NLSA parameters, out-of-sample data
        Out.nB           = 7;         % bathches to partition the in-sample data
        Out.nBRec        = Out.nB;    % batches for reconstructed data
        Out.nN           = 200;       % nearest neighbors for pairwise distances
        Out.lDist        = 'at';      % local distance
        Out.tol          = 0;         % 0 distance threshold (for cone kernel)
        Out.zeta         = In.zeta;   % cone kernel parameter    
        Out.coneAlpha    = In.coneAlpha; % velocity exponent in cone kernel
        Out.diffOpType   = 'gl';      % out-of-sample extension operator type
        Out.epsilon      = In.epsilon;% kernel bandwidth parameter 
        Out.alpha        = In.alpha;  % diffusion maps normalization 
        Out.nPhi         = In.nPhi;   % eigenfunctions to do OSE
        Out.idxPhiRecOSE = 1 : 5;     % eigenfunctions for OSE reconstruction
        Out.nNO          = Out.nN;    % nearest neighbors for OSE operator
        Out.epsilon      = In.epsilon;% kernel bandwidth parameter 
        Out.alpha        = In.alpha;  % Kernel normalization 
end


%% NLSA MODEL

%==============================================================================
% Fill in out-of sample parameters that can be determined by default from 
% the in-sample data
Out.nST       = In.nST;        % number of samples per period
Out.nEL       = In.nEL;        % delay embedding window length
Out.obsMap    = In.obsMap;     % observation function
Out.idxX      = In.idxX;       % data vector components to observe 
Out.fdOrder   = In.fdOrder;    % finite-difference order 
Out.fdType    = In.fdType;     % finite-difference type
Out.embFormat = In.embFormat;  % storage format for delay embedding

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

% Out-of-sample data 
Out.idxE  = In.idxE; % delay embedding indices
Out.idxT1 = Out.nEL + 1 + Out.nXB; % time origin for delay embedding
Out.nSSpin= Out.nST * Out.nTSpin; % number of samples used for spinup
Out.nS    = Out.nST * Out.nT + Out.nEL + Out.nXB + Out.nXA; % total # of samples
Out.dt    = 2 * pi / Out.nST; % sampling interval
Out.t     = linspace( 0, ( Out.nS - 1 ) * Out.dt, Out.nS ); % timestamps
Out.nSE   = Out.nS - Out.idxT1 + 1 - Out.nXA;% sample number after embedding

%==============================================================================
% Setup nlsaComponent objects for the in- and out-of-sample data

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

        strOut = [ 'r3' ...
                   '_r1'     num2str( Out.r1, '%1.2f' ) ...
                   '_r2'     num2str( Out.r2, '%1.2f' ) ...
                   '_f',     num2str( Out.f, '%1.2f' ) ...
                   '_aPhi'   num2str( Out.aPhi, '%1.2f' ) ...
                   '_aTheta' num2str( Out.aTheta, '%1.2f' ) ...
                   '_p'      num2str( Out.p ) ...
                   '_idxP'  sprintf( '%i_', Out.idxP ) ...
                   'dt'     num2str( Out.dt ) ...
                   '_nS'     int2str( Out.nS ) ...
                   '_nSSpin' int2str( Out.nSSpin ) ...
                   '_idxX'   sprintf( '%i_', Out.idxX ) ...  
                   'ifCent'  int2str( Out.ifCent ) ];

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

        strOut = [ 'r4' ...
                   '_f',     num2str( Out.f, '%1.2f' ) ...
                   '_aPhi'   num2str( Out.aPhi, '%1.2f' ) ...
                   '_aTheta' num2str( Out.aTheta, '%1.2f' ) ...
                   'dt'      num2str( Out.dt ) ...
                   '_nS'     int2str( Out.nS ) ...
                   '_nSSpin' int2str( Out.nSSpin ) ...
                   '_idxX'   sprintf( '%i_', Out.idxX ) ...  
                   'ifCent'  int2str( Out.ifCent ) ];
end

inPath   = fullfile( './data/raw',  strSrc ); % path for in-sample data
outPath  = fullfile( './data/raw',  strOut ); % path for out-of-sample data
nlsaPath = fullfile( './data/nlsa' );         % path for NLSA code output
tagSrc   = strSrc;                            % tag for in-sample data
tagOut   = strOut;                            % tag for out-of-sample data

nD  = numel( In.idxX );  % data space dimension for in-sample data
nDO = numel( Out.idxX ); % data space dimension for out-of-sample data

% Partition objects for the in-sample and out-of-sample data 
srcPartition    = nlsaPartition( 'nSample', In.nS );
outPartition    = nlsaPartition( 'nSample', Out.nS ); 
embPartition    = nlsaPartition( 'nSample', In.nSE, 'nBatch',  In.nB  );
outEmbPartition = nlsaPartition( 'nSample', Out.nSE, 'nBatch', Out.nB ); 

% Filename for in-sample data
% dataX.mat must contain an array x of size [ nD nS ], where
% nD is the dimension and nS the sample number
srcFilelist = nlsaFilelist( 'file', 'dataX.mat' );

% nlsaComponent object for in-sample data
srcComponent = nlsaComponent( 'partition',    srcPartition, ...
                              'dimension',    nD, ...
                              'path',         inPath, ...
                              'file',         srcFilelist, ...
                              'componentTag', tagSrc  );

% Filename for out-of-sample data
% dataX.mat must contain an array x of size [ nD nSO ]
outFilelist = nlsaFilelist( 'file', 'dataX.mat' );

% nlsaComponent object for out-of-sample data
outComponent = nlsaComponent( 'partition',    outPartition, ...
                              'dimension',    nDO, ...
                              'path',         outPath, ...
                              'file',         outFilelist, ...
                              'componentTag', tagOut );
                              


%==============================================================================
% Setup delay-embedding templates for the in- and out-of sample data

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

% Out-of-sample data
switch Out.embFormat
    case 'evector' % explicit delay-embedded vectors
        outEmbComponent = nlsaEmbeddedComponent_xi_e( 'idxE', Out.idxE, ...
                                                      'nXB',  Out.nXB, ...
                                                      'nXA', Out.nXA, ...
                                                      'fdOrder', Out.fdOrder,...
                                                      'fdType', Out.fdType );


     case 'overlap' % peform delay embedding on the fly
        outEmbComponent = nlsaEmbeddedComponent_xi_o( 'idxE', Out.idxE, ...
                                                      'nXB',  Out.nXB, ...
                                                      'nXA', Out.nXA, ...
                                                      'fdOrder', Out.fdOrder,...
                                                      'fdType', Out.fdType );
                                                          
end


%==============================================================================
% Pairwise distances for in-sample and out-of-sample data

% Pairwise distances for in-sample data
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

% Pairwise distances for out-of-sample data
switch Out.lDist
    case 'l2' % L^2 distance
        lDist = nlsaLocalDistance_l2();

    case 'at' % "autotuning" NLSA kernel
        lDist = nlsaLocalDistance_at(); 

    case 'cone' % cone kernel
        lDist = nlsaLocalDistance_cone( 'zeta', In.zeta, ...
                                        'tolerance', In.tol, ...
                                        'alpha', In.coneAlpha );
end
oseDFunc = nlsaLocalDistanceFunction( 'localDistance', lDist );
osePDist = nlsaPairwiseDistance( 'distanceFunction', oseDFunc, ...
                                 'nearestNeighbors', Out.nN );

%==============================================================================
% Symmetrized pairwise distances
sDist = nlsaSymmetricDistance_gl( 'nearestNeighbors', In.nNS );


%==============================================================================
% Diffusion operators 

% In-sample data
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

% Out-of-sample data
oseDiffOp = nlsaDiffusionOperator_ose( 'alpha',          Out.alpha, ...
                                       'epsilon',        Out.epsilon, ...
                                       'epsilonT',       In.epsilon, ...
                                       'nNeighbors',     Out.nNO, ...
                                       'nNeighborsT',    In.nNS, ...
                                       'nEigenfunction', Out.nPhi );
 
    
%==============================================================================
% Projections and linear map for SVD of the target data 
prjComponent = nlsaProjectedComponent( 'nBasisFunction', In.nPhiPrj );
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhiSVD );


%==============================================================================
% Reconstructed components

% Partitions
In.nSRec  = In.nSE + In.nEL;   % in-sample reconstructed data
Out.nSRec = Out.nSE + Out.nEL; % out-of-sample reconstructed data
recPartition = nlsaPartition( 'nSample', In.nSRec, ... 
                              'nBatch',  In.nBRec );
oseRecPartition = nlsaPartition( 'nSample', Out.nSRec, ...
                                 'nBatch',  Out.nBRec );

% Reconstructed data from diffusion eigenfnunctions
recComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxPhiRec );

% Reconstructed data from SVD 
svdRecComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxVTRec );

% Nystrom extension
oseEmbTemplate = nlsaEmbeddedComponent_ose_n( 'eigenfunctionIdx', Out.idxPhiRecOSE );
oseRecComponent = nlsaComponent_rec();

%==============================================================================
% Build NLSA model    
model = nlsaModel_ose( 'path',                            nlsaPath, ...
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
                       'svdReconstructionTemplate',       svdRecComponent, ...
                       'outComponent',                    outComponent, ...
                       'outTime',                         Out.t, ...
                       'outEmbeddingOrigin',              Out.idxT1, ...
                       'outEmbeddingTemplate',            outEmbComponent, ...
                       'outEmbeddingPartition',           outEmbPartition, ... 
                       'osePairwiseDistanceTemplate',     osePDist, ...
                       'oseDiffusionOperatorTemplate',    oseDiffOp, ...
                       'oseEmbeddingTemplate',            oseEmbTemplate, ...
                       'oseReconstructionPartition',      oseRecPartition );
                    
