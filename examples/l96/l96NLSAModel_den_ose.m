function [ model, In, Out ] = l96NLSAModel_den_ose( experiment )
%% L96LSAMODEL Build NLSA model with kernel density estimation and out-of-sample extension for the Lorenz 96 system. 
% 
%  In and Out are data structures containing the in-sample and out-of-sample model parameters, respectively.
%
%  See script l96Data.m for data generation.
%
%  For additional information on the arguments of nlsaModel( ... ) see 
%
%      nlsa/classes/nlsaModel_base/parseTemplates.m
%      nlsa/classes/nlsaModel/parseTemplates.m
%      nlsa/classes/nlsaModel_den/parseTemplates.m
%      nlsa/classes/nlsaModel_den_ose/parseTemplates.m     
%
% Modidied 2019/04/05
 
if nargin == 0
    experiment = 'test';
end

switch experiment

    case '128k'
        % In-sample dataset parameters
        In.n       = 41;            % number of nodes
        In.F       = 4;             % forcing parameter
        In.dt      = 0.01;          % sampling interval 
        In.idxX    = 1 : In.n;      % state vector observed components
        In.nSProd  = 128000;          % number of "production" samples
        In.nSSpin  = 512000;         % spinup samples
        In.nEL     = 0;             % embedding window (additional samples)
        In.nXB    = 1;     % additional samples before production interval (for FD)
        In.nXA    = 1;     % additional samples after production interval (for FD)
        In.x0      = 1;             % initial conditions
        In.relTol  = 1E-8;          % integrator tolerance
        In.ifCent  = false;         % data centering


        % Out-of-sample dataset parameters
        Out.n       = 41;            % number of nodes
        Out.F       = 4;             % forcing parameter
        Out.dt      = 0.01;          % sampling interval 
        Out.nSProd  = 64000;          % number of "production" samples
        Out.nSSpin  = 768000;         % spinup samples
        Out.nEL     = 0;             % embedding window (additional samples)
        Out.nXB    = 1;     % additional samples before production interval (for FD)
        Out.nXA    = 1;     % additional samples after production interval (for FD)
        Out.x0      = 1;             % initial conditions
        Out.relTol  = 1E-8;          % integrator tolerance
        Out.ifCent  = false;         % data centering
        Out.nS    =   Out.nSProd + Out.nEL + Out.nXB + Out.nXA; % sample number


        % NLSA parameters, in-sample data
        In.fdOrder      = 2;         % finite-difference order 
        In.fdType       = 'central'; % finite-difference type
        In.embFormat    = 'overlap'; % storage format for delay embedding
        In.nB           = 1;         % batches to partition the in-sample data
        In.nBRec        = In.nB;     % batches for reconstructed data
        In.nN           = 10000;     % nearest neighbors for pairwise distances
        In.lDist        = 'l2';      % local distance
        In.tol          = 0;         % 0 distance threshold (for cone kernel)
        In.zeta         = 0;         % cone kernel parameter 
        In.coneAlpha    = 0;         % velocity exponent in cone kernel
        In.nNS          = In.nN;     % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb_bs';      % diffusion operator type
        In.epsilon      = 1;         % kernel bandwidth parameter 
        In.epsilonB     = 2;         % kernel bandwidth base
        In.epsilonE     = [ -40 40 ];% kernel bandwidth exponents 
        In.nEpsilon     = 100;       % number of exponents for bandwidth tuning
        In.alpha        = .5;         % diffusion maps normalization 
        In.nPhi         = 1001;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;     % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;    % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 4;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 80;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -50 50 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 200;           % number of exponents for bandwidth tuning

        % NLSA parameters, out-of-sample data
        Out.nB           = 1;         % bathches to partition the in-sample data
        Out.nBRec        = Out.nB;    % batches for reconstructed data
        Out.nN           = In.nN;     % nearest neighbors for pairwise distances
        Out.lDist        = In.lDist;  % local distance
        Out.tol          = 0;         % 0 distance threshold (for cone kernel)
        Out.zeta         = In.zeta;   % cone kernel parameter    
        Out.coneAlpha    = In.coneAlpha; % velocity exponent in cone kernel
        Out.epsilon      = In.epsilon;% kernel bandwidth parameter 
        Out.alpha        = In.alpha;  % diffusion maps normalization 
        Out.nPhi         = In.nPhi;   % eigenfunctions to do OSE
        Out.idxPhiRecOSE = 1 : 5;     % eigenfunctions for OSE reconstruction
        Out.nNO          = Out.nN;    % nearest neighbors for OSE operator

        % NLSA parameters, out-of-sample KDE
        Out.denType     = In.denType;          % density estimation type
        Out.denND       = In.denND;             % manifold dimension for KDE
        Out.denLDist    = In.denLDist;          % local distance function for KDE
        Out.denBeta     = In.denBeta;  % density exponent 
        Out.denNN       = In.denNN;             % nearest neighbors for KDE
        Out.denZeta     = In.denZeta;             % cone kernel parameter (for KDE)
        Out.denAlpha    = In.denAlpha;             % cone kernel velocity exponent (for KDE)
        Out.denEpsilon = 1;

end


%% NLSA MODEL

%==============================================================================
% Fill in out-of sample parameters that can be determined by default from 
% the in-sample data
Out.fdOrder   = In.fdOrder;    % finite-difference order 
Out.fdType    = In.fdType;     % finite-difference type
Out.embFormat = In.embFormat;  % storage format for delay embedding
Out.idxX      = In.idxX;          % state vector observed components 

%==============================================================================
% Determine total number of samples, time origin, and delay-embedding indices

% In-sample data
In.idxE   = [ 1 : 1 : In.nEL + 1 ]; % delay embedding indices
In.idxT1  = In.nEL + 1 + In.nXB;    % time origin for delay embedding
In.nS     = In.nSProd + In.nEL + In.nXB + In.nXA; % total number of samples 
In.t      = linspace( 0, ( In.nS - 1 ) * In.dt, In.nS ); % timestamps
In.nSE    = In.nS - In.idxT1 + 1 - In.nXA; % number of samples after embedding

% Out-of-sample data 
Out.idxE  = In.idxE; % delay embedding indices
Out.idxT1 = Out.nEL + 1 + Out.nXB; % time origin for delay embedding
Out.t     = linspace( 0, ( Out.nS - 1 ) * Out.dt, Out.nS ); % timestamps
Out.nSE   = Out.nS - Out.idxT1 + 1 - Out.nXA;% sample number after embedding

%==============================================================================
% Fill initial conditions
In.x0  =  [ In.x0 zeros( 1, In.n - 1 ) ];
Out.x0 = [ Out.x0 zeros( 1, In.n - 1 ) ];

%==============================================================================
% Setup nlsaComponent objects for the in- and out-of-sample data 

strSrc = [ 'F'       num2str( In.F, '%1.3g' ) ...
           '_n'      int2str( In.n ) ... 
           '_dt'     num2str( In.dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', In.x0( 1  ) ) ...
           '_nS'     int2str( In.nS ) ...
           '_nSSpin' int2str( In.nSSpin ) ...
           '_relTol' num2str( In.relTol, '%1.3g' ) ...
           '_ifCent' int2str( In.ifCent ) ];

strOut = [ 'F'       num2str( Out.F, '%1.3g' ) ...
           '_n'      int2str( Out.n ) ... 
           '_dt'     num2str( Out.dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', Out.x0( 1  ) ) ...
           '_nS'     int2str( Out.nS ) ...
           '_nSSpin' int2str( Out.nSSpin ) ...
           '_relTol' num2str( Out.relTol, '%1.3g' ) ...
           '_ifCent' int2str( Out.ifCent ) ];

if numel( In.idxX ) == In.n && all( In.idxX == 1 : In.n )
    idxInfix = '';
else
    idxInfix = strcat( '_idxX', sprintf( '_%i', In.idxX ) );
end

inPath   = fullfile( './data/raw',  strSrc ); % path for in-sample data
outPath  = fullfile( './data/raw',  strOut ); % path for out-of-sample data
nlsaPath = fullfile( './data/nlsa' );         % path for NLSA code output
srcFilename = strcat( 'dataX', idxInfix, '.mat' ); % data filename 
tagSrc   = strSrc;                            % tag for in-sample data
tagOut   = strOut;                            % tag for out-of-sample data

nD  = In.n;  % data space dimension for in-sample data
nDO = nD;

% Partition objects for the in-sample and out-of-sample data 
srcPartition    = nlsaPartition( 'nSample', In.nS );
outPartition    = nlsaPartition( 'nSample', Out.nS ); 
embPartition    = nlsaPartition( 'nSample', In.nSE, 'nBatch',  In.nB  );
outEmbPartition = nlsaPartition( 'nSample', Out.nSE, 'nBatch', Out.nB ); 

% Filename for in-sample data
% dataX.mat must contain an array x of size [ nD nS ], where
% nD is the dimension and nS the sample number
srcFilelist = nlsaFilelist( 'file', srcFilename );

% nlsaComponent object for in-sample data
srcComponent = nlsaComponent( 'partition',    srcPartition, ...
                              'dimension',    nD, ...
                              'path',         inPath, ...
                              'file',         srcFilelist, ...
                              'componentTag', tagSrc  );

% Filename for out-of-sample data
% dataX.mat must contain an array x of size [ nD nSO ]
outFilelist = nlsaFilelist( 'file', srcFilename );

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
        if In.fdOrder ~= 0
            embComponent= nlsaEmbeddedComponent_xi_e( 'idxE', In.idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );
        else
            embComponent= nlsaEmbeddedComponent_e( 'idxE', In.idxE );
        end

    case 'overlap' % perform delay embedding on the fly
        if In.fdOrder ~= 0
            embComponent= nlsaEmbeddedComponent_xi_o( 'idxE', In.idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );
        else
            embComponent= nlsaEmbeddedComponent_o( 'idxE', In.idxE );
        end
end

% Out-of-sample data
switch Out.embFormat
    case 'evector' % explicit delay-embedded vectors
        if Out.fdOrder ~= 0
            outEmbComponent= nlsaEmbeddedComponent_xi_e( 'idxE', In.idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );
        else
            outEmbComponent= nlsaEmbeddedComponent_e( 'idxE', In.idxE );
        end

    case 'overlap' % perform delay embedding on the fly
        if Out.fdOrder ~= 0
            outEmbComponent= nlsaEmbeddedComponent_xi_o( 'idxE', In.idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );
        else
            outEmbComponent= nlsaEmbeddedComponent_o( 'idxE', In.idxE );
        end
end


%==============================================================================
% Pairwise distance for density estimation for in-sample and out-of sample data
%
% Pairwise distance for density estimation for in-sample data
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

% Pairwise distance for density estimation for out-of-sample
switch Out.denLDist
    case 'l2' % L^2 distance
        denLDist = nlsaLocalDistance_l2( 'mode', 'implicit' );

    case 'at' % "autotuning" NLSA kernel
        denLDist = nlsaLocalDistance_at( 'mode', 'implicit' );

    case 'cone' % cone kernel
        denLDist = nlsaLocalDistance_cone( 'mode', 'implicit', ...
                                           'zeta', Out.denZeta, ...
                                           'tolerance', Out.tol, ...
                                           'alpha', Out.denConeAlpha );
end

denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

oseDenPDist = nlsaPairwiseDistance( 'nearestNeighbors', Out.nN, ...
                                    'distanceFunction', denDFunc );

%==============================================================================
% Kernel density estimation for in-sample and out-of-sample data

% Kernel density estimation for in-sample data
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

% Kernel density estimation for out-of-sample data
switch Out.denType
    case 'fb' % fixed bandwidth
        oseDen = nlsaKernelDensity_ose_fb( ...
                 'dimension',              Out.denND, ...
                 'epsilon',                Out.denEpsilon );

    case 'vb' % variable bandwidth 
        oseDen = nlsaKernelDensity_ose_vb( ...
                 'dimension',              Out.denND, ...
                 'kNN',                    Out.denNN, ...
                 'epsilon',                Out.denEpsilon );
end



%==============================================================================
% Pairwise distance for density estimation for in-sample and out-of sample data
%
% Pairwise distance for density estimation for in-sample data
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

% Pairwise distance for density estimation for out-of-sample
switch Out.denLDist
    case 'l2' % L^2 distance
        denLDist = nlsaLocalDistance_l2( 'mode', 'implicit' );

    case 'at' % "autotuning" NLSA kernel
        denLDist = nlsaLocalDistance_at( 'mode', 'implicit' );

    case 'cone' % cone kernel
        denLDist = nlsaLocalDistance_cone( 'mode', 'implicit', ...
                                           'zeta', Out.denZeta, ...
                                           'tolerance', Out.tol, ...
                                           'alpha', Out.denConeAlpha );
end

denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

oseDenPDist = nlsaPairwiseDistance( 'nearestNeighbors', Out.nN, ...
                                    'distanceFunction', denDFunc );

%==============================================================================
% Kernel density estimation for in-sample and out-of-sample data

% Kernel density estimation for in-sample data
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

% Kernel density estimation for out-of-sample data
switch Out.denType
    case 'fb' % fixed bandwidth
        oseDen = nlsaKernelDensity_ose_fb( ...
                 'dimension',              Out.denND, ...
                 'epsilon',                Out.denEpsilon );

    case 'vb' % variable bandwidth 
        oseDen = nlsaKernelDensity_ose_vb( ...
                 'dimension',              Out.denND, ...
                 'kNN',                    Out.denNN, ...
                 'epsilon',                Out.denEpsilon );
end


%==============================================================================
% Pairwise distances for in-sample and out-of-sample data

% Pairwise distances for in-sample data
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
lScl = nlsaLocalScaling_pwr( 'pwr', 1 / In.denND );
dFunc = nlsaLocalDistanceFunction_scl( 'localDistance', lDist, ...
                                       'localScaling', lScl );
pDist = nlsaPairwiseDistance( 'distanceFunction', dFunc, ...
                              'nearestNeighbors', In.nN );

% Pairwise distances for out-of-sample data
switch Out.lDist
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

lScl = nlsaLocalScaling_pwr( 'pwr', 1 / Out.denND );
oseDFunc = nlsaLocalDistanceFunction_scl( 'localDistance', lDist, ...
                                          'localScaling', lScl );
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

    % global storage format, multiple bandwidth (automatic bandwidth selection and SVD)
    case 'gl_mb_svd' 
        diffOp = nlsaDiffusionOperator_gl_mb_svd( ...
                     'alpha',                  In.alpha, ...
                     'epsilon',                In.epsilon, ...
                     'nEigenfunction',         In.nPhi, ...
                     'bandwidthBase',          In.epsilonB, ...
                     'bandwidthExponentLimit', In.epsilonE, ...
                     'nBandwidth',             In.nEpsilon );

    case 'gl_mb_bs'
        diffOp = nlsaDiffusionOperator_gl_mb_bs( ...
                     'alpha',                  In.alpha, ...
                     'epsilon',                In.epsilon, ...
                     'nEigenfunction',         In.nPhi, ...
                     'bandwidthBase',          In.epsilonB, ...
                     'bandwidthExponentLimit', In.epsilonE, ...
                     'nBandwidth',             In.nEpsilon );

end

% Out-of-sample data
switch In.diffOpType
    case 'gl_mb_svd'
        oseDiffOp = nlsaDiffusionOperator_ose_svd( 'alpha',          Out.alpha, ...
                                       'epsilon',        Out.epsilon, ...
                                       'epsilonT',       In.epsilon, ...
                                       'nNeighbors',     Out.nNO, ...
                                       'nNeighborsT',    In.nNS, ...
                                       'nEigenfunction', Out.nPhi );
    case 'gl_mb_bs'
        oseDiffOp = nlsaDiffusionOperator_ose_bs( 'alpha',          Out.alpha, ...
                                       'epsilon',        Out.epsilon, ...
                                       'epsilonT',       In.epsilon, ...
                                       'nNeighbors',     Out.nNO, ...
                                       'nNeighborsT',    In.nNS, ...
                                       'nEigenfunction', Out.nPhi );

    otherwise
        oseDiffOp = nlsaDiffusionOperator_ose( 'alpha',          Out.alpha, ...
                                       'epsilon',        Out.epsilon, ...
                                       'epsilonT',       In.epsilon, ...
                                       'nNeighbors',     Out.nNO, ...
                                       'nNeighborsT',    In.nNS, ...
                                       'nEigenfunction', Out.nPhi );
end
 
    
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
model = nlsaModel_den_ose( 'path',                            nlsaPath, ...
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
                           'oseDenPairwiseDistanceTemplate',  oseDenPDist, ...
                           'oseKernelDensityTemplate',        oseDen, ...
                           'oseDiffusionOperatorTemplate',    oseDiffOp, ...
                           'oseEmbeddingTemplate',            oseEmbTemplate, ...
                           'oseReconstructionPartition',      oseRecPartition );
                    
