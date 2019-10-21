function [ model, In ] = torusNLSAModel_den( experiment )
%% TORUSNLSAMODEK_DEN Build NLSA model with kernel density estimation 
%  In are the in-sample parameters

switch experiment

    %% TEST CASE 
    case 'test'
        % In-sample dataset parameters
        In.f       = sqrt( 30 );
        In.aPhi    = 0.5;
        In.aTheta  = 0.5;
        In.r1      = .5;
        In.r2      = .5; 
        In.nST     = 50;   % samples per period
        In.nT      = 12;   % number of periods
        In.nEL     = 3;    % time-lagged embedding window length
        In.nXB     = 2;
        In.nXA     = 2;
        In.nS      = In.nST * In.nT + In.nEL + In.nXB + In.nXA;  % total number of samples
        In.idxX    = [ 1 : 3 ];
        In.p       = 0; % deformation
        In.idxP    = 3;
        In.ifCent  = false;

        
        % NLSA parameters
        In.fdOrder      = 4;
        In.fdType       = 'central';
        In.embFormat    = 'evector';
        In.idxE         = [ 1 : 1 : In.nEL + 1 ]; % time-lagged embedding indices
        In.nB           = 3;        % bathches to partition the source data
        In.nN           = 600;        % nearest neighbors for pairwise distances
        In.denND        = 2;
        In.denLDist     = 'l2';
        In.denBeta      = - 1 / In.denND;
        In.denNN        = 8;
        In.denZeta      = 0;
        In.denDistNorm  = 'geometric';
        In.denAlpha     = 0;
        In.lDist        = 'l2';
        In.tol          = 0;         % set pairwise distances to 0 below this threshold (for cone kernel)     
        In.zeta         = 0;     % cone kernel parameter    
        In.coneAlpha    = 0;
        In.distNorm     = 'geometric'; % kernel normalization (geometric/harmonic)
        In.nNS          = In.nN;        % nearest neighbors for symmetric distance
        In.nNMax        = round( 2 * In.nNS ); % for batch storage format
        In.epsilon      = 1;       % Gaussian Kernel width 
        In.alpha        = 1;         % Kernel normalization 
        In.nPhi         = 600;         % Laplace-Beltrami eigenfunctions
        In.nPhiPrj      = 600;
        In.idxPhi       = 1 : 10;    % eigenfunctions used for linear mapping
        In.idxPhiRec    = 1 : 10;


    %% MODEL I, CENTRAL 4TH-ORDER FD, CONE KERNEL 
    case 'modelI_64k'
        % In-sample dataset parameters
        In.f       = sqrt( 30 );
        In.aPhi    = 0.5;
        In.aTheta  = 0.5;
        In.r1      = .5;
        In.r2      = .5; 
        In.nST     = 500;  % samples per period
        In.nT      = 128;  % number of periods
        In.nEL     = 0;    % time-lagged embedding window length
        In.nXB     = 2;
        In.nXA     = 2;
        In.nS      = In.nST * In.nT + In.nEL + In.nXB + In.nXA;  % total number of samples
        In.idxX    = [ 1 : 3 ];
        In.p       = 0; % deformation
        In.idxP    = 3;
        In.ifCent  = false;

        
        % NLSA parameters
        In.fdOrder      = 4;
        In.fdType       = 'central';
        In.embFormat    = 'overlap';
        In.idxE         = [ 1 : 1 : In.nEL + 1 ]; % time-lagged embedding indices
        In.nB           = 32;        % bathches to partition the source data
        In.nN           = 5000;       % nearest neighbors for pairwise distances
        In.denND        = 2;
        In.denLDist     = 'cone';
        In.denBeta      = - 1 / In.denND;
        In.denNN        = 8;
        In.denZeta      = 0;
        In.denDistNorm  = 'geometric';
        In.denAlpha     = 0;
        In.lDist        = 'cone';
        In.tol          = 0;         % set pairwise distances to 0 below this threshold (for cone kernel)     
        In.zeta         = 0;     % cone kernel parameter    
        In.coneAlpha    = 0;
        In.distNorm     = 'geometric'; % kernel normalization (geometric/harmonic)
        In.nNS          = In.nN;        % nearest neighbors for symmetric distance
        In.nNMax        = round( 2 * In.nNS ); % for batch storage format
        In.epsilon      = 1;       % Gaussian Kernel width 
        In.alpha        = 1;         % Kernel normalization 
        In.nPhi         = 1001;       % Laplace-Beltrami eigenfunctions
        In.nPhiPrj      = 21;
        In.idxPhi       = 1 : 51;    % eigenfunctions used for linear mapping

end

%% NLSA MODEL
In.dt    = 2 * pi / In.nST / min( 1, In.f );
In.t     = linspace( 0, ( In.nS - 1 ) * In.dt, In.nS );
In.nE    = In.idxE( end );                          % embedding window       
In.nETrg = 0;
In.idxT1 = max( In.nE, In.nETrg ) + In.nXB;                 % time origin for embedding
In.nSE   = In.nS - In.idxT1 + 1 - In.nXA;                       % sample number after embedding

strSrc = [ 'r1'     num2str( In.r1, '%1.2f' ) ...
           '_r2'     num2str( In.r2, '%1.2f' ) ...
           '_f',     num2str( In.f, '%1.2f' ) ...
           '_aPhi'   num2str( In.aPhi, '%1.2f' ) ...
           '_aTheta' num2str( In.aTheta, '%1.2f' ) ...
           '_p'      num2str( In.p ) ...
           '_idxP'  sprintf( '%i_', In.idxP ) ...
           'dt'      num2str( In.dt ) ...
           '_nS'     int2str( In.nS ) ...
           '_idxX'   sprintf( '%i_', In.idxX ) ...  
           'ifCent'  int2str( In.ifCent ) ];

srcPath  = fullfile( './data/raw',  strSrc );
nlsaPath = fullfile( './data/nlsa' );
tagSrc   = strSrc;

nD    = numel( In.idxX );

% Source data assumed to be stored in a single batch, 
% embedded data in multiple batches
srcPartition    = nlsaPartition( 'nSample', In.nS );
embPartition    = nlsaPartition( 'nSample', In.nSE, 'nBatch',  In.nB  );

srcFilelist = nlsaFilelist( 'file', 'dataX.mat' );
srcComponent = nlsaComponent( 'partition', srcPartition, ...
                              'dimension', nD, ...
                              'path',      srcPath, ...
                              'file',      srcFilelist, ...
                              'componentTag', tagSrc  );
% dataX.mat must contain an array x of size [ nD nS ], where
% nD is the dimension and nS the sample number


switch In.embFormat
    case 'evector'
        embComponent= nlsaEmbeddedComponent_xi_e( 'idxE', In.idxE, ...
                                                  'nXB',  In.nXB, ...
                                                  'nXA', In.nXA, ...
                                                  'fdOrder', In.fdOrder, ...
                                                  'fdType', In.fdType );
    case 'overlap'
        embComponent= nlsaEmbeddedComponent_xi_o( 'idxE', In.idxE, ...
                                                  'nXB',  In.nXB, ...
                                                  'nXA', In.nXA, ...
                                                  'fdOrder', In.fdOrder, ...
                                                  'fdType', In.fdType );

end

prjComponent = nlsaProjectedComponent_xi( 'nBasisFunction', In.nPhiPrj );

denEmbComponent = nlsaEmbeddedComponent_e();

denPartition = nlsaPartition( 'nSample', In.nS, ...
                              'nBatch', In.nB );

% NLSA kernels -- source data
switch In.denLDist
    case 'l2'
        denLDist = nlsaLocalDistance_l2();

    case 'cone'
        denLDist = nlsaLocalDistance_cone( 'normalization', In.denDistNorm, ...
                                           'zeta', In.denZeta, ...
                                           'tolerance', In.tol, ...
                                           'alpha', In.denAlpha );

    case 'sone'
        denLDist = nlsaLocalDistance_sone( 'normalization', In.denDistNorm, ...
                                           'zeta', In.denZeta, ...
                                           'tolerance', In.tol, ...
                                           'ifVNorm', In.denIfVNorm );
end

switch In.lDist
    case 'l2'
        lScl  = nlsaLocalScaling_pwr( 'pwr', 1 / In.denND );
        lDist = nlsaLocalDistance_l2_scl( 'localScaling', lScl );

    case 'cone'
        lDist = nlsaLocalDistance_cone_den( 'normalization', In.distNorm, ...
                                            'zeta', In.zeta, ...
                                            'tolerance', In.tol, ...
                                            'alpha', In.coneAlpha, ...
                                            'beta', In.denBeta );

    case 'sone'
        lDist = nlsaLocalDistance_sone_den( 'normalization', In.distNorm, ...
                                            'zeta', In.zeta, ...
                                            'tolerance', In.tol, ...
                                            'ifVNorm', In.ifVNorm, ...
                                            'beta', In.denBeta );
end


dFunc = nlsaLocalDistanceFunction( 'localDistance', lDist );

pDist = nlsaPairwiseDistance( 'distanceFunction', dFunc, ...
                              'nearestNeighbors', In.nN );

sDist = nlsaSymmetricDistance_gl( 'nearestNeighbors', In.nNS ); 

diffOp = nlsaDiffusionOperator_gl_mb( 'alpha', In.alpha, ...
                                      'epsilon', In.epsilon, ...
                                      'nEigenfunction', In.nPhi );

denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

denPDist = nlsaPairwiseDistance( 'nearestNeighbors', In.nN, ...
                                 'distanceFunction', denDFunc );

den = nlsaKernelDensity_vb( 'dimension', In.denND, ...
                            'kNN',       In.denNN );

% Linear map
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhi );


recPartition = nlsaPartition( 'nSample', In.nSE + In.nE - 1, ...
                             'nBatch', 2 );
recComponent = nlsaComponent_rec( 'partition', recPartition, ...
                                  'basisFunctionIdx', In.idxPhiRec );
svdRecComponent = nlsaComponent_rec( 'partition', recPartition, ...
                                     'basisFunctionIdx', In.idxPhi );

% Build NLSA model    
model = nlsaModel_den( 'path',                            nlsaPath, ...
                       'srcTime',                         In.t, ...
                       'sourceComponent',                 srcComponent, ...
                       'embeddingOrigin',                 In.idxT1, ...
                       'embeddingTemplate',               embComponent, ...
                       'denEmbeddingOrigin',              1, ...
                       'denEmbeddingTemplate',            denEmbComponent, ...
                       'partitionTemplate',               embPartition, ...
                       'denPartitionTemplate',            denPartition, ...
                       'denPairwiseDistanceTemplate',     denPDist, ...
                       'kernelDensityTemplate',           den, ...
                       'densityEmbeddingTemplate',        embComponent, ...
                       'pairwiseDistanceTemplate',        pDist, ...
                       'symmetricDistanceTemplate',       sDist, ...
                       'diffusionOperatorTemplate',       diffOp, ...
                       'projectionTemplate',              prjComponent, ...
                       'reconstructionTemplate',          recComponent, ...
                       'linearMapTemplate',               linMap, ...
                       'svdReconstructionTemplate',       svdRecComponent );
                    
