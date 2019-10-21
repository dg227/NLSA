function [ model, In ] = monsoonNLSAModel( experiment )
switch experiment

    case 'std'

        % In-sample dataset parameters
        In.yrLim           = [ 2012 2013 ];       % time limits (years) 
        In.tFormat         = 'yyyydd';
        In.Src( 1 ).fld    = 'prec';
        In.Src( 1 ).xLim   = [];                  % longitude limitsi (empty if full domain)
        In.Src( 1 ).yLim   = [];                  % latitude limits 
        In.Src( 1 ).ifAnom = false;               % subtract climatology
        In.Src( 1 ).idxE   = [ 1 : 60 ];           % embedding indices
        In.Trg             = In.Src;

        % NLSA parameters
        In.nXB          = 1;
        In.nXA          = 1;
        In.fdOrder      = 2;
        In.fdType       = 'central';
        In.embFormat    = 'overlap';
        In.embFormatTrg = In.embFormat;

        In.nB           = 1;         % bathches to partition the source data
        In.nN           = 600;      % nearest neighbors for pairwise distances
        In.tol          = 0;         % set pairwise distances to 0 below this threshold (for cone kernel)     
        In.zeta         = 0;     % cone kernel parameter    
        In.ifVNorm      = false;
        In.distNorm     = 'geometric'; % kernel normalization (geometric/harmonic)
        In.nNS          = In.nN;        % nearest neighbors for symmetric distance
        In.nNMax        = round( 2 * In.nNS ); % for batch storage format
        In.epsilon      = 7;        % Gaussian Kernel width 
        In.alpha        = 1;        % Kernel normalization 
        In.nPhi         = 101;        % Laplace-Beltrami eigenfunctions
        In.nPhiPrj      = In.nPhi;
        In.idxPhi       = 1 : 15;    % eigenfunctions used for linear mapping

end

% Get timestamps from first source component
lbl = In.Src( 1 ).fld;
if In.Src( 1 ).ifAnom
    lbl = [ lbl '_anom' ];
end
xyr = sprintf( 'yr%i-%i', In.yrLim( 1 ), In.yrLim( 2 ) );
if ~isempty( In.Src( 1 ).xLim )
    xyr = [ xyr ...
            sprintf( '_x%i-%i_y%i-%i', In.Src( 1 ).xLim( 1 ), ...
                                       In.Src( 1 ).xLim( 2 ), ...
                                       In.Src( 1 ).yLim( 1 ), ...
                                       In.Src( 1 ).yLim( 2 ) ) ];
end
dataPath = fullfile( 'data/raw', lbl, xyr );
load( fullfile( dataPath, 'dataX.mat' ), 't' )
nS = numel( t ); % total number of samples

% Determine number of components and embedding indices
nC    = numel( In.Src );       % number of source components
nCTrg = numel( In.Trg );       % number of target components
nE    = In.Src( 1 ).idxE( end ); % max embedding index (source data)
for iC = 2 : nC
    nE = max( nE, In.Src( iC ).idxE( end ) );
end
nETrg  = In.Trg( 1 ).idxE( end ); % max embedding index (target data)
for iC = 2 : nCTrg
    nETrg = max( nETrg, In.Trg( iC ).idxE( end ) );
end
idxT1   = max( nE, nETrg ) + In.nXB; % time origin for embedding
nSE     = nS - idxT1 + 1 - In.nXA;   % sample number after embedding

% Source data assumed to be stored in a single batch, 
% embedded data in multiple batches
partition    = nlsaPartition( 'nSample', nS );
embPartition = nlsaPartition( 'nSample', nSE, 'nBatch',  In.nB  );

nlsaPath = 'data/nlsa';

% Determine source components
for iC = nC : -1 : 1

    lbl = In.Src( iC ).fld;
    if In.Src( iC ).ifAnom
        lbl = [ lbl '_anom' ];
    end
    xyr = sprintf( 'yr%i-%i', In.yrLim( 1 ), In.yrLim( 2 ) );
    if ~isempty( In.Src( iC ).xLim )
        xyr = [ xyr ...
                sprintf( '_x%i-%i_y%i-%i', In.Src( iC ).xLim( 1 ), ...
                                           In.Src( iC ).xLim( 2 ), ...
                                           In.Src( iC ).yLim( 1 ), ...
                                           In.Src( iC ).yLim( 2 ) ) ];
    end
    dataPath = fullfile( 'data/raw', lbl, xyr );
    tag = [ lbl '_' xyr ];

    % Data dimension
    load( fullfile( dataPath, 'dataGrid.mat' ), 'nD' )

    % dataX.mat must contain an array x of size [ nD nS ], where
    % nD is the dimension and nS the sample number
    fList = nlsaFilelist( 'file', 'dataX.mat' );

    srcComponent = nlsaComponent( 'partition', partition, ...
                                  'dimension', nD, ...
                                  'path',      dataPath, ...
                                  'file',      fList, ...
                                  'componentTag', tag  );

    switch In.embFormat
        case 'evector'
            embComponent= nlsaEmbeddedComponent_xi_e( 'idxE', In.Src( iC ).idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );
        case 'overlap'
            embComponent= nlsaEmbeddedComponent_xi_o( 'idxE', In.Src( iC ).idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );

    end
end


% Determine target components
for iC = nCTrg : -1 : 1

    lbl = In.Trg( iC ).fld;
    if In.Trg( iC ).ifAnom
        lbl = [ lbl '_anom' ];
    end
    xyr = sprintf( 'yr%i-%i', In.yrLim( 1 ), In.yrLim( 2 ) );
    if ~isempty( In.Trg( iC ).xLim )
        xyr = [ xyr ...
                sprintf( '_x%i-%i_y%i-%i', In.Trg( iC ).xLim( 1 ), ...
                                           In.Trg( iC ).xLim( 2 ), ...
                                           In.Trg( iC ).yLim( 1 ), ...
                                           In.Trg( iC ).yLim( 2 ) ) ];
    end
    dataPath = fullfile( 'data/raw', lbl, xyr );
    tag = [ lbl '_' xyr ];

    % Data dimension
    load( fullfile( dataPath, 'dataGrid.mat' ), 'nD' )

    % dataX.mat must contain an array x of size [ nD nS ], where
    % nD is the dimension and nS the sample number
    fList = nlsaFilelist( 'file', 'dataX.mat' );

    srcComponent = nlsaComponent( 'partition', partition, ...
                                  'dimension', nD, ...
                                  'path',      dataPath, ...
                                  'file',      fList, ...
                                  'componentTag', tag  );

    switch In.embFormat
        case 'evector'
            embComponent= nlsaEmbeddedComponent_xi_e( 'idxE', In.Trg( iC ).idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );
        case 'overlap'
            embComponent= nlsaEmbeddedComponent_xi_o( 'idxE', In.Trg( iC ).idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );

    end
end


prjComponent = nlsaProjectedComponent_xi( 'nBasisFunction', In.nPhiPrj );

% NLSA kernels -- source data


lDist = nlsaLocalDistance_l2( 'mode', 'implicit' );

%lDist = nlsaLocalDistance_cone( 'mode', 'implicit', ...
%                                'normalization', In.distNorm, ...
%                                'zeta', In.zeta, ...
%                                'tolerance', In.tol, ...
%                                'ifVNorm', In.ifVNorm );

pDist = nlsaPairwiseDistance( 'nearestNeighbors', In.nN, ...
                              'localDistance', lDist );

sDist = nlsaSymmetricDistance_gl( 'nearestNeighbors', In.nNS ); 

diffOp = nlsaDiffusionOperator_gl_mb( 'alpha', In.alpha, ...
                                      'epsilon', In.epsilon, ...
                                      'nEigenfunction', In.nPhi );  
        


% Array of linear maps with nested sets of basis functions

for iL = numel( In.idxPhi ) : -1 : 1
    linMap( iL ) = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhi( 1 : iL ) );
end


% Build NLSA model    
model = nlsaModel( 'path',                            nlsaPath, ...
                   'srcTime',                         t, ...
                   'timeFormat',                      In.tFormat, ...
                   'sourceComponent',                 srcComponent, ...
                   'embeddingOrigin',                 idxT1, ...
                   'embeddingTemplate',               embComponent, ...
                   'partitionTemplate',               embPartition, ...
                   'pairwiseDistanceTemplate',        pDist, ...
                   'symmetricDistanceTemplate',       sDist, ...
                   'diffusionOperatorTemplate',       diffOp, ...
                   'projectionTemplate',              prjComponent, ...
                   'linearMapTemplate',               linMap );
                    
% Output additional parameters
In.t     = t;
In.nS    = nS;
In.nSE   = nSE;
In.idxT1 = idxT1;
