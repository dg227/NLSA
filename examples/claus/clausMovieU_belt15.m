%% DATASET PARAMETERS
src        = 'tb_lat_av';                   % source data 
xLim       = [ 0 359.5  ];                  % longitude limits
yLim       = [ -15 15   ];                  % lattitude limits
nDX        = 1;                             % longitude downsampling
nDY        = 1;                             % lattitude downsampling
trg        = 'tb';                          % target data
xLimTrg    = [ 0 359.5 ];                   % lon. limit, target data
yLimTrg    = [ -15 15 ];                    % lat. limit, target data
nDXTrg     = 1;                             % x downsampling, target data
nDYTrg     = 1;                             % y downsampling, target data 
tLim       = { '1983070100' '2006063021' }; % time limits 
nDT        = 1;                             % downsampling in time

%% NLSA PARAMETERS
idxE         = 1 : 512; % time-lagged embedding indices
idxETrg      = 1 : 512; % embedding indices for target data
nB           = 8;       % bathches to partition the source data
nBTrg        = 8;       % batches to partition the target data
nN           = 5000;    % nearest neighbors for pairwise distances
nNS          = 5000;    % nearest neighbors for symmetric distance
epsilon      = 2;       % Gaussian Kernel width 
alpha        = 1;       % Kernel normalization 
nPhi         = 50;      % Laplace-Beltrami eigenfunctions
idxPhi       = 1 : 50;  % eigenfunctions used for linear mapping
filenameType = 'short'; % output file format


%% MOVIE PARAMETERS
idxA         = 25;                % linear map in the model
idxV         = [ 4 5 6 7 11 12 ];            % modes to display
nTileX       = 1;
nTileY       = 6;
cLimPat      = [ -.7 .7 ];
cTickPat     = [ -.7 : .1 : .1 ];
tagPos       = [ 48 12 ];
ifRead       = true;
visible      = 'on'; 

%% NLSA MODEL
tNum    = datenum( tLim{ 1 }, 'yyyymmddhh' ) ...
        : 1/8 ... 
        : datenum( tLim{ 2 }, 'yyyymmddhh' ); % timestamps
nS      = numel( tNum );                      % sample number
nE      = idxE( end );                        % embedding window          
nETrg   = idxETrg( end );                     % embedding window (target data)
t0      = max( nE, nETrg ) + 1;               % time origin for embedding
nSE     = nS - t0 + 1;                        % sample number after embedding

strSrc = [ 'x',    int2str( xLim( 1 ) ), '-', int2str( xLim( 2 ) ), ...
           '_y',   int2str( yLim( 1 ) ), '-', int2str( yLim( 2 ) ), ...
           '_',    tLim{ 1 }, '-', tLim{ 2 }, ...
           '_nDX', int2str( nDX ), ...
           '_nDY', int2str( nDY ), ...
           '_nDT', int2str( nDT ) ];

trgSrc = [ 'x',   int2str( xLimTrg( 1 ) ), '-', int2str( xLimTrg( 2 ) ), ...
           '_y',  int2str( yLimTrg( 1 ) ), '-', int2str( yLimTrg( 2 ) ), ...
           '_',  tLim{ 1 }, '-', tLim{ 2 }, ...
           '_nDX', int2str( nDXTrg ), ...
           '_nDY', int2str( nDYTrg ), ...
           '_nDT', int2str( nDT ) ];

srcPath  = fullfile( pwd, 'data/raw', src, strSrc );
trgPath  = fullfile( pwd, 'data/raw', trg, trgSrc );
nlsaPath = fullfile( pwd, 'data/nlsa' );
tagSrc   = [ src '_' strSrc ];
tagTrg   = [ trg '_' trgSrc ];

load( fullfile( trgPath, 'dataGrid.mat' ), 'nD' ); nDTrg = nD; % target data dimension
load( fullfile( srcPath, 'dataGrid.mat'),  'nD' );             % source data dimension

% Source and target data assumed to be stored in a single batch, 
% embedded data in multiple batches
srcPartition    = nlsaPartition( 'nSample', nS );
trgPartition    = nlsaPartition( 'nSample', nS ); 
embPartition    = nlsaPartition( 'nSample', nSE, 'nBatch', nB );
trgEmbPartition = nlsaPartition( 'nSample', nSE, 'nBatch', nBTrg ); 

srcComponent = nlsaComponent( 'partition', srcPartition, ...
                              'dimension', nD, ...
                              'path',      srcPath, ...
                              'file',      'dataAnomaly.mat', ...
                              'tag',       tagSrc );

trgComponent = nlsaComponent( 'partition', trgPartition, ...
                              'dimension', nDTrg, ...
                              'path',      trgPath, ...
                              'file',      'dataAnomaly.mat', ...
                              'tag',       tagTrg );

embComponent    = nlsaEmbeddedComponent_overlap( 'idxE', idxE, ...
                                                 'X0',   true );
trgEmbComponent = nlsaEmbeddedComponent_overlap( 'idxE', idxETrg, ...
                                                 'X0',   true );

pDist = nlsaPairwiseDistance_at( 'nearestNeighbors', nN );    % weights based on local scaling
%pDist = nlsaPairwiseDistance_gev( 'nearestNeighbors', nN );   % weights based on generalized eigenproblem

sDist= nlsaSymmetricDistance( 'nearestNeighbors', nNS ); % symmetrized weights

op = nlsaLaplacian( 'alpha', alpha, ...
                    'epsilon', epsilon, ...
                    'nEigenfunction', nPhi ); % Laplace-Beltrami operator

% Array of linear maps with nested sets of basis functions
for iL = numel( idxPhi ) : -1 : 1
    linMap( iL ) = nlsaLinearMap( 'idxEigenfunction', 1 : iL );
end

% Build NLSA model    
model = nlsaModel( 'path',                      nlsaPath, ...
                   'time',                      tNum, ...
                   'sourceComponent',           srcComponent, ...
                   'embeddingOrigin',           t0, ...
                   'embeddingTemplate',         embComponent, ...
                   'partitionTemplate',         embPartition, ...
                   'pairwiseDistanceTemplate',  pDist, ...
                   'symmetricDistanceTemplate', sDist, ...
                   'temporalOperatorTemplate',  op, ...
                   'targetComponent',           trgComponent, ...
                   'targetEmbeddingOrigin',     t0, ...
                   'targetEmbeddingTemplate',   trgEmbComponent, ...
                   'targetPartitionTemplate',   trgEmbPartition, ...
                   'linearMapTemplate',         linMap, ...
                   'filenameType',              filenameType ); 


%% MOVIE DATA
load( fullfile( trgPath, 'dataGrid.mat' ), 'x', 'y', 'w', 'nX', 'nY', 'ifDef', 'nD' ); 
if ifRead
    op = getLinearMap( model, idxA );
    u  = getLSVectors( op );
    nV = numel( idxV );
    nE = numel( idxE );
    U  = cell( 1, nV );
    for iV = 1 : nV
        U{ iV } = reshape( u( :, idxV( iV ) ), [ nD nE ] );
        U{ iV } = bsxfun( @rdivide, U{ iV }, w' );
        U{ iV } = U{ iV } / max( abs( U{ iV }( : ) ) );
    end
end



%% MAKE MOVIE
movieDir = makeMovieDir( model, idxA );
if ~isdir( movieDir )
    mkdir( movieDir )
end

movieFile = [ movieDir,'/', ...
              'movieU', ...
              '_idxV',   sprintf( '_%i', idxV ), ...
              '.avi' ];
          
convFact = 100;
panelX   = 7 * convFact;
panelY   = panelX * (3 / 4)^8;
deltaX   = 0.5 * convFact;
deltaX2  = 0.1 * convFact;
deltaY   = 0.3 * convFact;
deltaY2  = 0.25 * convFact;
gapX     = 0.1 * convFact;
gapY     = 0.1 * convFact;

posn    = [ 0, ...
            0, ...
            nTileX * panelX + ( nTileX - 1 ) * gapX + deltaX + deltaX2, ...
            nTileY * panelY + ( nTileY - 1 ) * gapY + deltaY + deltaY2 ];

aviObj   = avifile( movieFile, 'fps', 8 );

fig = figure( 'units', 'pixels', ...
              'position', posn, ...
              'visible', visible, ...
              'color', 'white', ...
              'doubleBuffer', 'on', ...
              'backingStore', 'off', ...
              'defaultAxesTickDir', 'out', ...
              'defaultAxesNextPlot', 'replace', ...
              'defaultAxesBox', 'on', ...
              'defaultAxesFontSize', 10, ...
              'defaultTextFontSize', 8, ...
              'defaultAxesTickDir',  'out', ...
              'defaultAxesTickLength', [ 0.006 0 ], ...
              'defaultAxesFontName', 'lucida console', ...
              'defaultTextFontName', 'lucida console', ...
              'defaultAxesLineWidth', 1, ...
              'defaultAxesLayer', 'top', ...
              'renderer', 'zbuffer' );

ax    = zeros( nTileX, nTileY );
axPos = cell( nTileX, nTileY );

iV = 1;
for jAx = 1 : nTileY 
    for iAx = 1 : nTileX
        posnY = deltaY + ( nTileY - jAx ) * ( panelY + gapY );
        axPos{ iAx, jAx } =  [ deltaX + ( iAx - 1 ) * ( panelX + gapX ), posnY, panelX, panelY ];
        if iV <= nV
            ax( iAx, jAx ) = axes( 'units', 'pixels', ...
                                   'position', [ deltaX + ( iAx - 1 ) * ( panelX + gapX ), ...
                                    posnY, panelX, panelY ] );
        end
        iV = iV + 1;
    end
end              

axTitle = axes( 'units', 'pixels', 'position', [ deltaX, deltaY, ...
                              nTileX * panelX + ( nTileX - 1 ) * gapX, ...
                              nTileY * panelY + ( nTileY - 1 ) * gapY ], ...
                'color', 'none', 'box', 'off' );

cData            = zeros( nY, nX ); 
cPos             = cell( size( ax ) );

for iE = 1 : nE
    disp( [ 'Lag ', int2str( iE ), '/', int2str( nE ) ] )        

    for iV = 1 : nV
        
        cData( ifDef ) = U{ iV }( :, iE );
                	  
        set( gcf, 'currentAxes', ax( iV ) )
        
        m_proj( 'equidistant cylindrical','lat', yLim, 'long', xLim );
        h = m_pcolor( x, y, cData );
        set( h, 'edgeColor', 'none' )
        [ subPat( 1 ) subPat( 2 ) ] = ind2sub( [ nTileX nTileY ], iV );
        if subPat( 1 ) == 1 && subPat( 2 ) ~= nTileY
            m_grid( 'linest', 'none', ...
                    'linewidth', 1 , ...
                    'tickdir','out', ...
                    'xTick', [ xLim( 1 ) : 20 : floor( xLim( 2 ) ) ], ...
                    'yTick', [ yLim( 1 ) : 5 : yLim( 2 ) ], ...
                    'xTickLabels', [] );
        elseif subPat( 1 ) ~= 1 && subPat( 2 ) == nTileY
            m_grid( 'linest', 'none', ...
                    'linewidth', 1 , ...
                    'tickdir','out', ...
                    'xTick', [ xLim( 1 ) : 20 : floor( xLim( 2 ) ) ], ...
                    'yTick', [ yLim( 1 ) : 5 : yLim( 2 ) ], ...
                    'yTickLabels', [] );
        elseif subPat( 1 ) == 1 && subPat( 2 ) == nTileY
            m_grid( 'linest', 'none', ...
                    'linewidth', 1 , ...
                    'tickdir','out', ...
                    'xTick', [ xLim( 1 ) : 20 : floor( xLim( 2 ) ) ], ...
                    'yTick', [ yLim( 1 ) : 5 : yLim( 2 ) ] );
        else
            m_grid( 'linest', 'none', ...
                    'linewidth', 1 , ...
                    'tickdir','out', ...
                    'xTick', [ xLim( 1 ) : 20 : floor( xLim( 2 ) ) ], ...
                    'yTick', [ yLim( 1 ) : 5 : yLim( 2 ) ], ...
                    'xTickLabels', [], ...
                    'yTickLabels', [] );
        end     
        m_coast( 'line', 'linewidth', 1, 'color', 'k' );
        tgOff = 0;
        [ X, Y ] = m_ll2xy( tagPos( 1 ), tagPos( 2 ) + tgOff );
        text( X, Y, int2str( idxV( iV ) ) )
        %hC     = colorbar( 'eastOutside', 'peer', ax( axPat{ iV } ) );
        %if iE == 1
        %    cPos{ axPat{ iV } }   = get( hC, 'position' );
        %    cPos{ axPat{ iV } }( 1 ) = cPos{ axPat{ iV } }( 1 ) + 0.045;
        %    cPos{ axPat{ iV } }( 3 ) = cPos{ axPat{ iV } }( 3 ) * .3;
        % end
        set( gca, 'cLim', [ -.7 .7 ], 'position', axPos{ iV } )
        %set( hC, 'position', cPos{ axPat{ iV } }, 'yTick', cTickPat{ iV } )
                
    end
	
    iHrLag = ( nE - iE ) * 3;
    iDLag  = floor( iHrLag / 24 );
    iHrLag = iHrLag - iDLag * 24; 
    title( axTitle,  [ 'lag ', int2str( iDLag ), ' d ', int2str( iHrLag ), ' h'  ] )
    axis( axTitle, 'off' )

    set( fig, 'position', posn )
    drawnow
    aviObj = addframe( aviObj, fig );
    for iAx = 1 : numel( ax )
       cla( ax( iAx ), 'reset' )
    end
    cla( axTitle, 'reset' )

end

aviObj = close( aviObj );
