%% LORENZ 96 MODEL
%  
%  Incompressible velocity field in 2D

%% DATASET PARAMETERS
n      = 40;     % number of nodes
F      = 4;      % forcing parameter
dt     = 0.01;  % sampling interval 
nSProd = 4000;   % number of "production" samples
nSSpin = 16000;  % spinup samples
nEL    = 0;      % embedding window length (additional samples)
nXB    = 0;      % additional samples before production interval (for FD)
nXA    = 0;      % additional samples after production interval (for FD)
x0      = zeros( 1, n );
x0( 1 ) = 1;
relTol = 1E-8;
ifCent = false;        % data centering
lX = 2 * pi; % x domain size
lY = 2 * pi; % y domain size

%% PLOT PARAMETERS
idxPlotU = [ 1 128 1024 2048 ];


%% MOVIE PARAMETERS
Mov.figWidth   = 300;    % in pixels
Mov.deltaX     = 35;
Mov.deltaX2    = 40;
Mov.deltaY     = 30;
Mov.deltaY2    = 15;
Mov.visible    = 'off';
fps        = 24;
idxMovieU = 1 : 512; 


%% SCRIPT OPTIONS
ifReadS      = true;  % read state
ifCalcU      = true;  % calculate velocity field
ifWriteUNLSA = true; % write velocity field in NLSA format
ifPlotU      = false; 
ifMovieU     = false;

nS = nSProd + nEL + nXB + nXA;
strSrc = [ 'F'       num2str( F, '%1.3g' ) ...
           '_dt'     num2str( dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', x0( 1  ) ) ...
           '_nS'     int2str( nS ) ...
           '_nSSpin' int2str( nSSpin ) ...
           '_relTol' num2str( relTol, '%1.3g' ) ...
           '_ifCent' int2str( ifCent ) ];


%% READ DATA
if ifReadS
    pathSrc = fullfile( './data', 'raw', strSrc );
    filename = fullfile( pathSrc, 'dataX.mat' );
    load( filename, 'x' )
    s = x;
    clear x
end

%% COMPUTE VELOCITY FIELD
if ifCalcU
    %% COMPUTE VELOCITY FIELD
    [ u, v, x, y, c ] = state2vel( s, lX, lY );
end


%% VELOCITY FIELD SNAPSHOTS
if ifPlotU
    for iPlt = 1 : numel( idxPlotU )
        figure;
        quiver( x, y, squeeze( u( :, :, idxPlotU( iPlt ) ) ), ...
                      squeeze( v( :, :, idxPlotU( iPlt ) ) ) )
        set( gca, 'xLim', [ 0 lX ], 'yLim', [ 0 lY ] )
    end
end 

%% VELOCITY FIELD MOVIES
if ifMovieU

    % Determine figure sizes
    nTileX = 1;
    nTileY = 1;
    gapX   = 0;
    gapY   = 0;
    panelX = ( Mov.figWidth - Mov.deltaX - Mov.deltaX2 - ( nTileX -1 ) * gapX ) / nTileX;
    panelY = panelX;

    posn     = [ 0, ...
                 0, ...
                 nTileX * panelX + ( nTileX - 1 ) * gapX + Mov.deltaX + Mov.deltaX2, ...
                 nTileY * panelY + ( nTileY - 1 ) * gapY + Mov.deltaY + Mov.deltaY2 ];


    % Determine movie directoru and filename
    movieDir = fullfile( './movies', strSrc );
    strMov = [ 'lX'    num2str( lX, '%1.3g' ), ...
               '_lY'   num2str( lY, '%1.3g' ), ...
               '_idxT' int2str( idxMovieU( 1 ) ) '-' int2str( idxMovieU( end ) ) ];
    if ~isdir( movieDir )
        mkdir( movieDir )
    end
    movieFile = fullfile( movieDir, [ strMov '.avi' ] );

    % Make movie
    writerObj = VideoWriter( movieFile );
    writerObj.FrameRate = fps;
    open( writerObj );

    fig = figure( 'units', 'pixels', ...
              'paperunits', 'points', ...
              'position', posn, ...
              'paperPosition', posn, ...
              'visible', Mov.visible, ...
              'color', 'white', ...
              'doubleBuffer', 'on', ...
              'backingStore', 'off', ...
              'defaultAxesTickDir', 'out', ...
              'defaultAxesNextPlot', 'replace', ...
              'defaultAxesBox', 'on', ...
              'defaultAxesFontSize', 8, ...
              'defaultTextFontSize', 8, ...
              'defaultAxesTickDir',  'out', ...
              'defaultAxesTickLength', [ 0.01 0 ], ...
              'defaultAxesFontName', 'helvetica', ...
              'defaultTextFontName', 'helvetica', ...
              'defaultAxesLineWidth', 1, ...
              'defaultAxesLayer', 'top' );

    ax = zeros( nTileX, nTileY );

    for iAx = 1 : nTileX
        for jAx = 1 : nTileY
            ax( iAx, jAx ) = axes( ...
                'units', 'pixels', ...
                'position', ...
                [ Mov.deltaX + ( iAx - 1 ) * ( panelX + gapX ), ...
                  Mov.deltaY + ( nTileY - jAx ) * ( panelY + gapY ), ...
                  panelX, panelY ] );
        end
    end
              
    nT = numel( idxMovieU );
    for iT = 1 : nT
        disp( sprintf( 'Frame %i/%i', iT, nT ) );
        quiver( x, y, squeeze( u( :, :, idxMovieU( iT ) ) ), ...
                      squeeze( v( :, :, idxMovieU( iT ) ) ) )
        title( sprintf( 't = %1.3f', t( idxMovieU( iT ) ) ) )
        set( gca, 'xLim', [ 0 lX ], 'yLim', [ 0 lY ] )
        frame = export_fig(fig, '-nocrop', '-pdf', '-a1', '-r100');
        writeVideo( writerObj, im2frame(frame) );
        cla( gca, 'reset' )

    end
    close( writerObj );
    close( fig );   
end


