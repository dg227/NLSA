function [ model, In ] = monsoonNLSAModel( experiment )
switch experiment

    case 'std'

        % In-sample dataset parameters
        In.yrLim           = [ 2005 2012 ];       % time limits (years) 
        In.dateLim         = { '0501' '1001' };
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
        In.epsilon      = 2;        % Gaussian Kernel width 
        In.alpha        = 1;        % Kernel normalization 
        In.nPhi         = 101;        % Laplace-Beltrami eigenfunctions
        In.nPhiPrj      = In.nPhi;
        In.idxPhi       = 1 : 15;    % eigenfunctions used for linear mapping

end

nlsaPath = 'data/nlsa';

lbl = In.Src( 1 ).fld;
if In.Src( 1 ).ifAnom
    lbl = [ lbl '_anom' ];
end

dateStr{ 1 } = [ In.dateLim{ 1 } int2str( In.yrLim( 1 ) ) ];
dateStr{ 2 } = [ In.dateLim{ 2 } int2str( In.yrLim( 2 ) ) ];
dateStrFull = strjoin( dateStr, '-' );

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

nR = In.yrLim( 2 ) - In.yrLim( 1 ) + 1;
tR = cell( 1, nR );

for iR = nR : -1 : 1

    yr = In.yrLim( 1 ) + iR - 1;
    dateStr{ 1 } = [ In.dateLim{ 1 } int2str( yr ) ];
    dateStr{ 2 } = [ In.dateLim{ 2 } int2str( yr ) ];

    for iC = nC : -1 : 1
        xyr = strjoin( dateStr, '-' );
        if ~isempty( In.Src( iC ).xLim )
            xyr = [ xyr ...
                    sprintf( '_x%i-%i_y%i-%i', In.Src( iC ).xLim( 1 ), ...
                                               In.Src( iC ).xLim( 2 ), ...
                                               In.Src( iC ).yLim( 1 ), ...
                                               In.Src( iC ).yLim( 2 ) ) ];
        end
        %tag = [ lbl '_' xyr ];
        dataPath = fullfile( 'data/raw', lbl, xyr );

        if iC == nC
            load( fullfile( dataPath, 'dataX.mat' ), 't' )
            nS  = numel( t ); % total number of samples
            nSE = nS - idxT1 + 1 - In.nXA;   % sample number after embedding
            tNum{ iR } = t;
    
            % Source data assumed to be stored in a single batch, 
            % embedded data in multiple batches
            partition          = nlsaPartition( 'nSample', nS );
            embPartition( iR ) = nlsaPartition( 'nSample', nSE, 'nBatch',  In.nB  );

        end

        if iR == nR
            % Data dimension
            load( fullfile( dataPath, 'dataGrid.mat' ), 'nD' )
        end

        % dataX.mat must contain an array x of size [ nD nS ], where
        % nD is the dimension and nS the sample number
        fList = nlsaFilelist( 'file', 'dataX.mat' );

        srcComponent( iC, iR ) = nlsaComponent( 'partition', partition, ...
                                                'dimension', nD, ...
                                                'path',      dataPath, ...
                                                'file',      fList, ...
                                                'componentTag', lbl, ...
                                                'realizationTag', xyr  );
    end
end

for iC = nC : -1 : 1
    switch In.embFormat
        case 'evector'
            embComponent( iC ) = nlsaEmbeddedComponent_xi_e( 'idxE', In.Src( iC ).idxE, ...
                                                      'nXB',  In.nXB, ...
                                                      'nXA', In.nXA, ...
                                                      'fdOrder', In.fdOrder, ...
                                                      'fdType', In.fdType );
        case 'overlap'
            embComponent( iC ) = nlsaEmbeddedComponent_xi_o( 'idxE', In.Src( iC ).idxE, ...
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
                   'sourceRealizationName',           dateStrFull, ...  
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
