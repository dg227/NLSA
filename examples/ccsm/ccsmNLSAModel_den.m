function [ model, In ] = ccsmNLSAModel_den( experiment )
%% CCSMNLSAMODEL_DEN Build NLSA model with kernel density estimation for  
%  monthly data from the CCSM/CESM climate models. 
% 
%  In is a data structure containing the model parameters (named after 
%  "in-sample," as opposed to "out-of-sample" data).
%
%  See script ccsmImport.m for additional details on the dynamical system.
%
%  For additional information on the arguments of nlsaModel( ... ) see 
%
%      ../classes/nlsaModel_base/parseTemplates.m
%      ../classes/nlsaModel/parseTemplates.m
%      ../classes/nlsaModel_den/parseTemplates.m
%
% Modidied 2016/03/20

switch experiment

    %% NORTH PACIFIC SST
    case 'np_sst'

        % In-sample dataset parameters 
        In.tFormat             = 'yyyymm';    % time format
        In.Res( 1 ).yrLim      = [ 1 1300 ];  % time limits (in years) for realization 1 
        In.Res( 1 ).experiment = 'b40.1850';  % CCSM4/CESM experiment
        In.Src( 1 ).field      = 'SST';       % field for source component 1
        In.Src( 1 ).xLim       = [ 120 250 ]; % longitude limits
        In.Src( 1 ).yLim       = [ 20  65  ]; % latitude limits
        In.Trg( 1 ).field      = 'SST';       % gield for target component 1
        In.Trg( 1 ).xLim       = [ 120 250 ]; % longitude limits
        In.Trg( 1 ).yLim       = [ 20  65  ]; % latitude limits

        % NLSA parameters
        In.Src( 1 ).idxE    = 1 : 24;      % delay embedding indices for source component 1 
        In.Src( 1 ).fdOrder = 2;           % finite-difference order 
        In.Src( 1 ).fdType  = 'central';   % finite-difference type
        In.Src( 1 ).embFormat = 'overlap'; % storage format for delay embedding
        In.Trg( 1 ).idxE    = 1 : 24;   % delay embedding indices for target component 1
        In.Trg( 1 ).fdOrder = 2;         % finite-difference order 
        In.Trg( 1 ).fdType  = 'central'; % finite-difference type
        In.Trg( 1 ).embFormat = 'overlap'; % storage format for delay embedding

        In.Res( 1 ).nXB = 1;         % samples to leave out before main interval
        In.Res( 1 ).nXA = 1;         % samples to leave out after main interval
        In.Res( 1 ).nB  = 64;        % batches to partition the in-sample data (realization 1)
        In.Res( 1 ).nBRec = In.Res.nB; % batches for reconstructed data
        In.nN           = 15300;      % nearest neighbors for pairwise distances
        In.lDist        = 'at';       % local distance
        In.tol          = 0;          % 0 distance threshold (for cone kernel)
        In.zeta         = 0.995;      % cone kernel parameter 
        In.coneAlpha    = 1;          % velocity exponent in cone kernel
        In.nNS          = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';    % diffusion operator type
        In.epsilon      = 2;          % kernel bandwidth parameter 
        In.epsilonB     = 2;          % kernel bandwidth base
        In.epsilonE     = [ -40 40 ]; % kernel bandwidth exponents 
        In.nEpsilon     = 200;        % number of exponents for bandwidth tuning
        In.alpha        = 1;          % diffusion maps normalization 
        In.nPhi         = 51;         % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;    % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;      % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;     % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;      % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 6;             % manifold dimension for KDE
        In.denLDist    = 'l2';          % local distance function for KDE
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors for KDE
        In.denZeta     = 0;             % cone kernel parameter (for KDE)
        In.denAlpha    = 0;             % cone kernel velocity exponent (for KDE)
        In.denEpsilonB = 2;             % kernel bandwidth base (for KDE)
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents (for KDE)
        In.denNEpsilon = 200;           % number of exponents for bandwidth tuning

end

%% NLSA MODEL

nlsaPath = './data/nlsa';

%==============================================================================
% Determine total number of samples, time origin for delay embedding

In.nC      = numel( In.Src );    % number of source components
In.nCT     = numel( In.Trg ); % number of target compoents
In.nR      = numel( In.Res ); % number of realizations

for iR = In.nR : -1 : 1
    In.Res( iR ).nYr  = In.Res( iR ).yrLim( 2 ) - In.Res( iR ).yrLim( 1 ) + 1; % number of years
    In.Res( iR ).nS   = 12 * In.Res( iR ).nYr; % number of monthly samples
    In.Res( iR ).tNum = zeros( 1, In.Res( iR ).nS );     % standard Matlab timestamps
    iS   = 1;
    for iYr = 1 : In.Res( iR ).nYr
        for iM = 1 : 12
            In.Res( iR ).tNum( iS ) = datenum( sprintf( '%04i%02i', In.Res( iR ).yrLim( 1 ) + iYr - 1, iM  ), 'yyyymm' );
            iS         = iS + 1;
        end
    end
end

In.nE = In.Src( 1 ).idxE( end ); % maximum number of delay embedding lags for source data
for iC = 2 : In.nC
    In.nE = max( In.nE, In.Src( iC ).idxE( end ) );
end
In.nET  = In.Trg( 1 ).idxE( end ); % maximum number of delay embedding lags for target data
for iC = 2 : In.nCT
    In.nET = max( In.nET, In.Trg( iC ).idxE( end ) );
end
In.idxT1   = max( In.nE, In.nET ) + In.Res.nXB;     % time origin for delay embedding
In.Res.nSE = In.Res.nS - In.idxT1 + 1 - In.Res.nXA; % sample number after embedding

%==============================================================================
% Setup nlsaComponent and nlsaProjectedComponent objects 

fList = nlsaFilelist( 'file', 'dataX.mat' ); % filename for source data

% Loop over realizations
for iR = In.nR : -1 : 1

    yr = sprintf( 'yr%i-%i', In.Res( iR ).yrLim( 1 ), In.Res( iR ).yrLim( 2 ) );
    tagR = [ In.Res( 1 ).experiment '_' yr ];
    partition = nlsaPartition( 'nSample', In.Res( iR ).nS ); % source data assumed to be stored in a single batch
    embPartition( iR ) = nlsaPartition( 'nSample', In.Res( iR ).nSE, ...
                                        'nBatch',  In.Res( iR ).nB  );

    % Loop over source components
    for iC = In.nC : -1 : 1

        xy = sprintf( 'x%i-%i_y%i-%i', In.Src( iC ).xLim( 1 ), ...
                                       In.Src( iC ).xLim( 2 ), ...
                                       In.Src( iC ).yLim( 1 ), ...
                                       In.Src( iC ).yLim( 2 ) );

        pathC = fullfile( './data/raw/',  ...
                         In.Res( iR ).experiment, ...
                         In.Src( iC ).field,  ...
                         [ xy '_' yr ] );
                                                   
        tagC = [ In.Src( iC ).field '_' xy ];

        load( fullfile( pathC, 'dataGrid.mat' ), 'nD' ) % data space dimension
        In.Src( iC ).nD = nD;

        srcComponent( iC, iR ) = nlsaComponent( 'partition',      partition, ...
                                                'dimension',      In.Src( iC ).nD, ...
                                                'path',           pathC, ...
                                                'file',           fList, ...
                                                'componentTag',   tagC, ...
                                                'realizationTag', tagR  );

        switch In.Src( iC ).embFormat
            case 'evector'
                embComponent( iC, iR )= nlsaEmbeddedComponent_xi_e( ...
                                        'idxE',    In.Src( iC ).idxE, ... 
                                        'nXB',     In.Res( iR ).nXB, ...
                                        'nXA',     In.Res( iR ).nXA, ...
                                        'fdOrder', In.Src( iC ).fdOrder, ...
                                        'fdType',  In.Src( iC ).fdType );
            case 'overlap'
                embComponent( iC, iR) = nlsaEmbeddedComponent_xi_o( ...
                                        'idxE',    In.Src( iC ).idxE, ...
                                        'nXB',     In.Res( iR ).nXB, ...
                                        'nXA',     In.Res( iR ).nXA, ...
                                        'fdOrder', In.Src( iC ).fdOrder, ...
                                        'fdType',  In.Src( iC ).fdType );
        end
    
    end

    % Loop over target components
    for iC = In.nCT : -1 : 1

        xy = sprintf( 'x%i-%i_y%i-%i', In.Trg( iC ).xLim( 1 ), ...
                                       In.Trg( iC ).xLim( 2 ), ...
                                       In.Trg( iC ).yLim( 1 ), ...
                                       In.Trg( iC ).yLim( 2 ) );

        pathC = fullfile( './data/raw/',  ...
                         In.Res( iR ).experiment, ...
                         In.Trg( iC ).field,  ...
                         [ xy '_' yr ] );
                                                   
        tagC = [ In.Trg( iC ).field '_' xy ];

        load( fullfile( pathC, 'dataGrid.mat' ), 'nD' ) % data space dimension
        In.Trg( iC ).nD = nD;

        trgComponent( iC, iR ) = nlsaComponent( 'partition',      partition, ...
                                                'dimension',      In.Trg( iC ).nD, ...
                                                'path',           pathC, ...
                                                'file',           fList, ...
                                                'componentTag',   tagC, ...
                                                'realizationTag', tagR  );

        switch In.Trg( iC ).embFormat
            case 'evector'
                trgEmbComponent( iC, iR )= nlsaEmbeddedComponent_xi_e( ...
                                      'idxE',    In.Trg( iC ).idxE, ... 
                                      'nXB',     In.Res( iR ).nXB, ...
                                      'nXA',     In.Res( iR ).nXA, ...
                                      'fdOrder', In.Trg( iC ).fdOrder, ...
                                      'fdType',  In.Trg( iC ).fdType );
            case 'overlap'
                trgEmbComponent( iC, iR) = nlsaEmbeddedComponent_xi_o( ...
                                           'idxE',    In.Trg( iC ).idxE, ...
                                           'nXB',     In.Res( iR ).nXB, ...
                                           'nXA',     In.Res( iR ).nXA, ...
                                           'fdOrder', In.Trg( iC ).fdOrder, ...
                                           'fdType',  In.Trg( iC ).fdType );
        end


        prjComponent( iC ) = nlsaProjectedComponent_xi( ...
                                 'nBasisFunction', In.nPhiPrj );
    end
end


        

%=================================================================================
% Pairwise distance for density estimation
switch In.Src( 1 ).embFormat
    case 'evector'
        modeStr = 'explicit';
    case 'overlap'
        modeStr = 'implicit';
end

switch In.denLDist
    case 'l2' % L^2 distance
        denLDist = nlsaLocalDistance_l2( 'mode', modeStr );

    case 'at' % "autotuning" NLSA kernel
        denLDist = nlsaLocalDistance_at( 'mode', modeStr ); 

    case 'cone' % cone kernel
        denLDist = nlsaLocalDistance_cone( 'mode', modeStr, ...
                                           'normalization', In.denDistNorm, ...
                                           'zeta', In.denZeta, ...
                                           'tolerance', In.tol, ...
                                           'alpha', In.denAlpha );
end

denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

denPDist = nlsaPairwiseDistance( 'nearestNeighbors', In.nN, ...
                                 'distanceFunction', denDFunc );
%================================================================================
%Kernel density estimation

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
% Pairwise distances for in-sample data
switch In.lDist
    case 'l2' % L^2 distance
        lDist = nlsaLocalDistance_l2( 'mode', modeStr );

    case 'at' % "autotuning" NLSA kernel
        lDist = nlsaLocalDistance_at( 'mode', modeStr );

    case 'cone' % cone kernel
        lDist = nlsaLocalDistance_cone( 'mode', modeStr, ...
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
        diffOp = nlsaDiffusionOperator_gl(    'alpha',          In.alpha, ...
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
% Linear map for SVD of the target data 
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhiSVD );

%==============================================================================
% Reconstructed components
In.Res.nSRec  = In.Res.nSE + In.nE - 1; % number of reconstructed samples
for iR = In.nR : -1 : 1

    % Partition for reconstructed data
    recPartition = nlsaPartition( 'nSample', In.Res( iR ).nSRec, ... 
                                  'nBatch',  In.Res( iR ).nBRec );

    % Reconstructed data from diffusion eigenfnunctions
    recComponent( 1, iR ) = nlsaComponent_rec_phi( 'partition', recPartition, ...
                                                   'basisFunctionIdx', In.idxPhiRec );

    % Reconstructed data from SVD 
    svdRecComponent( 1, iR ) = nlsaComponent_rec_phi( 'partition', recPartition, ...
                                                      'basisFunctionIdx', In.idxVTRec );
end

%==============================================================================
% Build NLSA model    
model = nlsaModel_den( 'path',                            nlsaPath, ...
                       'timeFormat',                      In.tFormat, ...
                       'srcTime',                         { In.Res.tNum }, ...
                       'sourceComponent',                 srcComponent, ...
                       'embeddingOrigin',                 In.idxT1, ...
                       'embeddingTemplate',               embComponent, ...
                       'embeddingPartition',              embPartition, ...
                       'denPairwiseDistanceTemplate',     denPDist, ...
                       'kernelDensityTemplate',           den, ...
                       'pairwiseDistanceTemplate',        pDist, ...
                       'symmetricDistanceTemplate',       sDist, ...
                       'diffusionOperatorTemplate',       diffOp, ...
                       'targetComponent',                 trgComponent, ...
                       'targetEmbeddingTemplate',         trgEmbComponent, ...
                       'projectionTemplate',              prjComponent, ...
                       'reconstructionTemplate',          recComponent, ...
                       'linearMapTemplate',               linMap, ...
                       'svdReconstructionTemplate',       svdRecComponent );

