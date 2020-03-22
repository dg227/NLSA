function [ model, In ] = climateNLSAModel_den( experiment )
%% CLIMATENLSAMODEL_DEN NLSA model with kernel density estimation for climate
%  datasets
% 
%  In is a data structures containing the in-sample model parameters
%
%  The following scripts are provided for data import: 
%
%      hadisstData.m: HadISST1 dataset  
%      noaaData.m:    NOAA 20th Century Reanalysis dataset
%
%  For additional information on the arguments of nlsaModel_den( ... ) see:
%
%      nlsa/classes/nlsaModel_base/parseTemplates.m
%      nlsa/classes/nlsaModel/parseTemplates.m
%      nlsa/classes/nlsaModel_den/parseTemplates.m
% 
% Structure fields Src represent different physical variables (e.g., SST) 
% employed in the kernel definition
%
% Structure fields Res represent different realizations (ensemble members)
%
% For HadISST1: Longitude range is [ 0.5 359.5 ] at 1 degree increments
%               Latitude range is [ -89.5 89.5 ] at 1 degree increements
%               Date range is Jan 1870 to Feb 2019 at 1 month increments 
%               
% For NOAA: Longitude range is [ 0 359 ] at 1 degree increments
%           Latitude range is [ -89 89 ] at 1 degree increments
%           Date range is is Jan 1854 to Feb 2019  at 1 month increments
%
% For GPCP1DDv1.2: Longitude range is [ 0.5 359.5 ] at 1 degree increments
%                  Latitude range is [ -89.5 89.5 ] at 1 degree increements
%                  Date range is 01 Oct 1996 to 31 Oct 2015 at 1 day increments 
%
% Modified 2019/07/31
 
if nargin == 0
    experiment = 'ip_sst'; % Indo-Pacific SST
end

switch experiment

    % INDO-PACIFIC SST
    case 'ip_sst'

        % In-sample dataset parameters 
        % Source (covariate) data is area-weighted Indo-Pacific SST
        % Target (response) data is Nino 3.4 index
        In.tFormat             = 'yyyymm';    % time format
        In.freq                = 'daily';       % sampling frequency
        In.Res( 1 ).tLim       = { '187001' '198712' }; % time limit  
        In.Res( 1 ).experiment = 'noaa';      % NOAA dataset
        In.Src( 1 ).field      = 'sstmaw_198101-201012';       % physical field
        In.Src( 1 ).xLim       = [ 28 290 ];  % longitude limits
        In.Src( 1 ).yLim       = [ -60  20 ]; % latitude limits
        In.Trg( 1 ).field      = 'sstmawav_198101-201012';      % physical field
        In.Trg( 1 ).xLim       = [ 190 240 ];  % longitude limits
        In.Trg( 1 ).yLim       = [ -5 5 ]; % latitude limits

        % NLSA parameters, in-sample data
        In.Src( 1 ).idxE      = 1 : 24;    % delay embedding indices 
        In.Src( 1 ).nXB       = 1;   % samples to leave out before main interval
        In.Src( 1 ).nXA       = 23;  % samples to leave out after main interval
        In.Src( 1 ).fdOrder   = 1;         % finite-difference order 
        In.Src( 1 ).fdType    = 'backward'; % finite-difference type
        In.Src( 1 ).embFormat = 'overlap'; % storage format for delay embedding
        In.Trg( 1 ).idxE      = 1 : 1;     % delay embedding indices 
        In.Trg( 1 ).nXB       = 1;   % samples to leave out before main interval
        In.Trg( 1 ).nXA       = 23;  % samples to leave out after main interval
        In.Trg( 1 ).fdOrder   = 1;         % finite-difference order 
        In.Trg( 1 ).fdType    = 'central'; % finite-difference type
        In.Trg( 1 ).embFormat = 'overlap'; % storage format for delay embedding
        In.Res( 1 ).nB        = 1;   % partition batches
        In.Res( 1 ).nBRec     = 1; % batches for reconstructed data
        In.nN         = 0;   % nearest neighbors; defaults to max. value if 0
        In.nN         = 0;   % nearest neighbors; defaults to max. value if 0
        In.lDist      = 'l2';   % local distance
        In.tol        = 0;      % 0 distance threshold (for cone kernel)
        In.zeta       = 0.995;  % cone kernel parameter 
        In.coneAlpha  = 1;      % velocity exponent in cone kernel
        In.nNS        = In.nN;  % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb'; % diffusion operator type
        In.epsilon    = 1;          % kernel bandwidth parameter 
        In.epsilonB   = 2;          % kernel bandwidth base
        In.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
        In.nEpsilon   = 200;      % number of exponents for bandwidth tuning
        In.alpha      = 0;        % diffusion maps normalization 
        In.nPhi       = 1001;     % diffusion eigenfunctions to compute
        In.nPhiPrj    = In.nPhi;  % eigenfunctions to project the data
        In.idxPhiRec  = 1 : 1;    % eigenfunctions for reconstruction
        In.idxPhiSVD  = 1 : 1;    % eigenfunctions for linear mapping
        In.idxVTRec   = 1 : 1;    % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 5;             % manifold dimension 
        In.denLDist    = 'l2';          % local distance function 
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 50;            % nearest neighbors 
        In.denZeta     = 0;             % cone kernel parameter 
        In.denConeAlpha= 0;             % cone kernel velocity exponent 
        In.denEpsilon  = 1;             % kernel bandwidth
        In.denEpsilonB = 2;             % kernel bandwidth base 
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents 
        In.denNEpsilon = 200;        % number of exponents for bandwidth tuning

    % SOUTH ASIAN SUMMER MONSOON PRECIP
    case 'monsoon_precip'

        % In-sample dataset parameters 
        % Source (covariate) data is South Asian summer Monsoon precip
        % Target (response) data is South Asian summer Monsoon precip
        In.tFormat             = 'yyyymmdd';    % time format
        In.freq                = 'daily';       % sampling frequency
        In.Res( 1 ).tLim       = { '19961001' '20101231' }; % time limit  
        In.Res( 1 ).experiment = 'gpcp_1dd_v1.2';      % GPCP dataset
        % South Asian summer Monsoon domain 
        In.Src( 1 ).field      = 'precip';       % physical field
        In.Src( 1 ).xLim       = [ 30 160 ];  % longitude limits
        In.Src( 1 ).yLim       = [ -20 40 ]; % latitude limits
        % South Asian summer Monsoon domain
        In.Trg( 1 ).field      = 'precip';      % physical field
        In.Trg( 1 ).xLim       = [ 30 160 ];  % longitude limits
        In.Trg( 1 ).yLim       = [ -20 40 ]; % latitude limits

        % NLSA parameters, in-sample data
        % South Asian summer Monsoon domain
        In.Src( 1 ).idxE      = 1 : 64;    % delay embedding indices 
        In.Src( 1 ).nXB       = 1;   % samples to leave out before main interval
        In.Src( 1 ).nXA       = 0;  % samples to leave out after main interval
        In.Src( 1 ).fdOrder   = 1;         % finite-difference order 
        In.Src( 1 ).fdType    = 'backward'; % finite-difference type
        In.Src( 1 ).embFormat = 'overlap'; % storage format for delay embedding
        % South Asian summer Monsoon domain
        In.Trg( 1 ).idxE      = 1 : 1;     % delay embedding indices 
        In.Trg( 1 ).nXB       = 1;   % samples to leave out before main interval
        In.Trg( 1 ).nXA       = 0;  % samples to leave out after main interval
        In.Trg( 1 ).fdOrder   = 1;         % finite-difference order 
        In.Trg( 1 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 1 ).embFormat = 'overlap'; % storage format for delay embedding
        In.Res( 1 ).nB        = 1;   % partition batches
        In.Res( 1 ).nBRec     = 1; % batches for reconstructed data
        In.nN         = 0;   % nearest neighbors; defaults to max. value if 0
        In.nN         = 0;   % nearest neighbors; defaults to max. value if 0
        In.lDist      = 'l2';   % local distance
        In.tol        = 0;      % 0 distance threshold (for cone kernel)
        In.zeta       = 0.995;  % cone kernel parameter 
        In.coneAlpha  = 1;      % velocity exponent in cone kernel
        In.nNS        = In.nN;  % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon    = 1;          % kernel bandwidth parameter 
        In.epsilonB   = 2;          % kernel bandwidth base
        In.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
        In.nEpsilon   = 200;      % number of exponents for bandwidth tuning
        In.alpha      = 0.5;        % diffusion maps normalization 
        In.nPhi       = 101;     % diffusion eigenfunctions to compute
        In.nPhiPrj    = In.nPhi;  % eigenfunctions to project the data
        In.idxPhiRec  = 1 : 1;    % eigenfunctions for reconstruction
        In.idxPhiSVD  = 1 : 1;    % eigenfunctions for linear mapping
        In.idxVTRec   = 1 : 1;    % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 5;             % manifold dimension 
        In.denLDist    = 'l2';          % local distance function 
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 50;            % nearest neighbors 
        In.denZeta     = 0;             % cone kernel parameter 
        In.denConeAlpha= 0;             % cone kernel velocity exponent 
        In.denEpsilon  = 1;             % kernel bandwidth
        In.denEpsilonB = 2;             % kernel bandwidth base 
        In.denEpsilonE = [ -40 40 ];    % kernel bandwidth exponents 
        In.denNEpsilon = 200;        % number of exponents for bandwidth tuning

end


%% ROOT DIRECTORY NAMES
inPath   = fullfile( pwd, 'data/raw' );  % in-sample data
nlsaPath = fullfile( pwd, 'data/nlsa' ); % nlsa output


%% DELAY-EMBEDDING ORIGINGS
In.nC  = numel( In.Src ); % number of source components
In.nCT = numel( In.Trg ); % number of target compoents

% Maximum number of delay embedding lags, sample left out in the 
% beginning/end of the analysis interval for source data
In.nE = In.Src( 1 ).idxE( end ); 
In.nXB = In.Src( 1 ).nXB; 
In.nXA = In.Src( 1 ).nXA;
for iC = 2 : In.nC
    In.nE = max( In.nE, In.Src( iC ).idxE( end ) );
    In.nXB = max( In.nXB, In.Src( iC ).nXB );
    In.nXA = max( In.nXA, In.Src( iC ).nXA );
end

% Maximum number of delay embedding lags, sample left out in the 
% beginning/end of the analysis interval for targe data
nETMin  = In.Trg( 1 ).idxE( end ); % minimum number of delays for target data
In.nET  = In.Trg( 1 ).idxE( end ); % maximum number of delays for target data
In.nXBT = In.Trg( 1 ).nXB;
In.nXAT = In.Trg( 1 ).nXA;
for iC = 2 : In.nCT
    In.nET = max( In.nET, In.Trg( iC ).idxE( end ) );
    nETMin = min( In.nET, In.Trg( iC ).idxE( end ) );
    In.nXBT = min( In.nXBT, In.Trg( iC ).nXB );
    In.nXAT = min( In.nXAT, In.Trg( iC ).nXA );
end
nEMax = max( In.nE, In.nET );
nXBMax = max( In.nXB, In.nXBT );
nXAMax = max( In.nXA, In.nXAT );

%% NUMBER OF STAMPLES AND TIMESTAMPS FOR IN-SAMPLE DATA
In.nR  = numel( In.Res ); % number of realizations, in-sample data
% Number of samples and timestaamps for in-sample data
nSETot = 0;
idxT1 = zeros( 1, In.nR );
tNum = cell( 1, In.nR ); % Matlab serial date numbers
for iR = In.nR : -1 : 1
    limNum = datenum( In.Res( iR ).tLim, In.tFormat );
    switch In.freq
    case 'daily'
        tNum{ iR } = limNum( 1 ) : limNum( 2 ); 
        In.Res( iR ).nS = numel( tNum{ iR } ); % number of daily samples
    case 'monhtly'
        In.Res( iR ).nS   = months( limNum( 1 ), limNum( 2 ) ) + 1; % number of monthly samples
        tNum{ iR } = datemnth( limNum( 1 ), 0 : In.Res( iR ).nS - 1 ); 
    end
    In.Res( iR ).idxT1 = nEMax + nXBMax;     % time origin for delay embedding
    idxT1( iR ) = In.Res( iR ).idxT1;
    In.Res( iR ).nSE = In.Res( iR ).nS - In.Res( iR ).idxT1 + 1 - nXAMax; % sample number after embedding
    nSETot = nSETot + In.Res( iR ).nSE;
    In.Res( iR ).nSRec = In.Res( iR ).nSE + nETMin - 1; % sample number for reconstruction 
end
if In.nN == 0
   In.nN = nSETot;
end 
if In.nNS == 0
    In.nNS = nSETot;
end

%% IN-SAMPLE DATA COMPONENTS
fList = nlsaFilelist( 'file', 'dataX.mat' ); % filename for source data

% Loop over realizations for in-sample data
for iR = In.nR : -1 : 1

    tStr = [ In.Res( iR ).tLim{ 1 } '-' In.Res( iR ).tLim{ 2 } ]; 
    tagR = [ In.Res( iR ).experiment '_' tStr ];
                                    
    partition = nlsaPartition( 'nSample', In.Res( iR ).nS ); % source data assumed to be stored in a single batch
    embPartition( iR ) = nlsaPartition( 'nSample', In.Res( iR ).nSE, ...
                                        'nBatch',  In.Res( iR ).nB  );
    recPartition( iR ) = nlsaPartition( 'nSample', In.Res( iR ).nSRec, ...
                                        'nBatch',  In.Res( iR ).nBRec );

    % Loop over source components
    for iC = In.nC : -1 : 1

        xyStr = sprintf( 'x%i-%i_y%i-%i', In.Src( iC ).xLim( 1 ), ...
                                          In.Src( iC ).xLim( 2 ), ...
                                          In.Src( iC ).yLim( 1 ), ...
                                          In.Src( iC ).yLim( 2 ) );

        pathC = fullfile( inPath,  ...
                          In.Res( iR ).experiment, ...
                          In.Src( iC ).field,  ...
                          [ xyStr '_' tStr ] );
                                                   
        tagC = [ In.Src( iC ).field '_' xyStr ];

        load( fullfile( pathC, 'dataGrid.mat' ), 'nD' )
        
        srcComponent( iC, iR ) = nlsaComponent( 'partition',      partition, ...
                                                'dimension',      nD, ...
                                                'path',           pathC, ...
                                                'file',           fList, ...
                                                'componentTag',   tagC, ...
                                                'realizationTag', tagR  );

    end

    % Loop over target components 
    for iC = In.nCT : -1 : 1

        xyStr = sprintf( 'x%i-%i_y%i-%i', In.Trg( iC ).xLim( 1 ), ...
                                          In.Trg( iC ).xLim( 2 ), ...
                                          In.Trg( iC ).yLim( 1 ), ...
                                          In.Trg( iC ).yLim( 2 ) );

        pathC = fullfile( inPath,  ...
                          In.Res( iR ).experiment, ...
                          In.Trg( iC ).field,  ...
                          [ xyStr '_' tStr ] );
                                                   
        tagC = [ In.Trg( iC ).field '_' tStr ];


        load( fullfile( pathC, 'dataGrid.mat' ), 'nD'  )

        trgComponent( iC, iR ) = nlsaComponent( 'partition',      partition, ...
                                                'dimension',      nD, ...
                                                'path',           pathC, ...
                                                'file',           fList, ...
                                                'componentTag',   tagC, ...
                                                'realizationTag', tagR  );
    end

end

% Loop over source components to create embedding templates
for iC = In.nC : -1 : 1
    switch In.Src( iC ).embFormat
        case 'evector'
            if In.Src( iC ).fdOrder < 0
                embComponent( iC, 1 ) = nlsaEmbeddedComponent( ...
                                    'idxE',    In.Src( iC ).idxE, ... 
                                    'nXB',     In.Src( iC ).nXB, ...
                                    'nXA',     In.Src( iC ).nXA );
            else
                embComponent( iC, 1 )= nlsaEmbeddedComponent_xi_e( ...
                                    'idxE',    In.Src( iC ).idxE, ... 
                                    'nXB',     In.Src( iC ).nXB, ...
                                    'nXA',     In.Src( iC ).nXA, ...
                                    'fdOrder', In.Src( iC ).fdOrder, ...
                                    'fdType',  In.Src( iC ).fdType );
            end
        case 'overlap'
            if In.Src( iC ).fdOrder < 0
                embComponent( iC, 1 ) = nlsaEmbeddedComponent_o( ...
                                    'idxE',    In.Src( iC ).idxE, ...
                                    'nXB',     In.Src( iC ).nXB, ...
                                    'nXA',     In.Src( iC ).nXA );
            else
                embComponent( iC, 1 ) = nlsaEmbeddedComponent_xi_o( ...
                                    'idxE',    In.Src( iC ).idxE, ...
                                    'nXB',     In.Src( iC ).nXB, ...
                                    'nXA',     In.Src( iC ).nXA, ...
                                    'fdOrder', In.Src( iC ).fdOrder, ...
                                    'fdType',  In.Src( iC ).fdType );
            end
    end
end

% Loop over target components to create embedding templates
for iC = In.nCT : -1 : 1
    switch In.Trg( iC ).embFormat
        case 'evector'
            if In.Trg( iC ).fdOrder < 0
                trgEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_e( ...
                                      'idxE',    In.Trg( iC ).idxE, ... 
                                      'nXB',     In.Trg( iC ).nXB, ...
                                      'nXA',     In.Trg( iC ).nXA );
            else
                trgEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_xi_e( ...
                                      'idxE',    In.Trg( iC ).idxE, ... 
                                      'nXB',     In.Trg( iC ).nXB, ...
                                      'nXA',     In.Trg( iC ).nXA, ...
                                      'fdOrder', In.Trg( iC ).fdOrder, ...
                                      'fdType',  In.Trg( iC ).fdType );
             end
        case 'overlap'
            if In.Trg( iC ).fdOrder < 0 
                trgEmbComponent( iC, iC ) = nlsaEmbeddedComponent_o( ...
                                       'idxE',    In.Trg( iC ).idxE, ...
                                       'nXB',     In.Trg( iC ).nXB, ...
                                       'nXA',     In.Trg( iC ).nXA );
            else
                trgEmbComponent( iC, iC ) = nlsaEmbeddedComponent_xi_o( ...
                                       'idxE',    In.Trg( iC ).idxE, ...
                                       'nXB',     In.Trg( iC ).nXB, ...
                                       'nXA',     In.Trg( iC ).nXA, ...
                                       'fdOrder', In.Trg( iC ).fdOrder, ...
                                       'fdType',  In.Trg( iC ).fdType );
            end
    end
end


%% PROJECTED COMPONENTS
for iC = In.nCT : -1 : 1
    if isa( trgEmbComponent( iC, 1 ), 'nlsaEmbeddedComponent_xi' )
        prjComponent( iC ) = nlsaProjectedComponent_xi( ...
                             'nBasisFunction', In.nPhiPrj );
    else
        prjComponent( iC ) = nlsaProjectedComponent( ...
                             'nBasisFunction', In.nPhiPrj );
    end
end


%% PAIRWISE DISTANCE FOR DENSITY ESTIMATION FOR IN-SAMPLE DATA
if all( strcmp( { In.Src.embFormat }, 'overlap' ) )
    modeStr = 'implicit';
else
    modeStr = 'explicit';
end

switch In.denLDist
    case 'l2' % L^2 distance
        denLDist = nlsaLocalDistance_l2( 'mode', modeStr );

    case 'at' % "autotuning" NLSA kernel
        denLDist = nlsaLocalDistance_at( 'mode', modeStr );

    case 'cone' % cone kernel
        denLDist = nlsaLocalDistance_cone( 'mode', modeStr, ...
                                           'zeta', In.denZeta, ...
                                           'tolerance', In.tol, ...
                                           'alpha', In.denConeAlpha );
end

denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

denPDist = nlsaPairwiseDistance( 'nearestNeighbors', In.nN, ...
                                 'distanceFunction', denDFunc );

denPDist = repmat( denPDist, [ In.nC 1 ] );



%% KERNEL DENSITY ESTIMATION FOR IN-SAMPLE DATA
switch In.denType
    case 'fb' % fixed bandwidth
        den = nlsaKernelDensity_fb( ...
                 'dimension',              In.denND, ...
                 'epsilon',                In.denEpsilon, ...
                 'bandwidthBase',          In.denEpsilonB, ...
                 'bandwidthExponentLimit', In.denEpsilonE, ...
                 'nBandwidth',             In.denNEpsilon );

    case 'vb' % variable bandwidth 
        den = nlsaKernelDensity_vb( ...
                 'dimension',              In.denND, ...
                 'epsilon',                In.denEpsilon, ...
                 'kNN',                    In.denNN, ...
                 'bandwidthBase',          In.denEpsilonB, ...
                 'bandwidthExponentLimit', In.denEpsilonE, ...
                 'nBandwidth',             In.denNEpsilon );
end

den = repmat( den, [ In.nC 1 ] );

%% PAIRWISE DISTANCES FOR IN-SAMPLE DATA
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
lScl = nlsaLocalScaling_pwr( 'pwr', 1 / In.denND );
dFunc = nlsaLocalDistanceFunction_scl( 'localDistance', lDist, ...
                                       'localScaling', lScl );
pDist = nlsaPairwiseDistance( 'distanceFunction', dFunc, ...
                              'nearestNeighbors', In.nN );

%% SYMMETRIZED PAIRWISE DISTANCES
sDist = nlsaSymmetricDistance_gl( 'nearestNeighbors', In.nNS );


%% DIFFUSION OPERATOR FOR IN-SAMPLE DATA
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

%% LINEAR MAP FOR SVD OF TARGET DATA
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhiSVD );


%% RECONSTRUCTED TARGET COMPONENTS -- IN-SAMPLE DATA
% Reconstructed data from diffusion eigenfnunctions
recComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxPhiRec );

% Reconstructed data from SVD 
svdRecComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxVTRec );


%% BUILD NLSA MODEL    
model = nlsaModel_den( 'path',                            nlsaPath, ...
                       'sourceComponent',                 srcComponent, ...
                       'targetComponent',                 trgComponent, ...
                       'srcTime',                         tNum, ...
                       'embeddingOrigin',                 idxT1, ...
                       'embeddingTemplate',               embComponent, ...
                       'targetEmbeddingTemplate',         trgEmbComponent, ...
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
                       'svdReconstructionTemplate',       svdRecComponent );
                    
