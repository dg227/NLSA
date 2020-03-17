%% PLOT MOVIE OF KERNEL DENSITY AND EIGENFUNCTIONS 
%
% Input data: Sea ice concentration and ocean velocity

%% DATASET/NLSA PARAMETERS
experiment = 'channel_cuvocn'; 

xLim =  [ -2.5E5 2.5E5 ]; 
yLim =  [ -5E4 5E4 ]; 
nX = 25;  % number of coarse cells, x direction
nY = 5;   % number of coarse cells, y direction

dt = 20;  % time step (sec)
nDT = 50; % number of timesteps between output

idxE = 1; % index within embedding window to plot raw data.

idxPhi = [ 2 : 16 ]; % eigenfunctions to plot

%% SCRIPT OPTIONS
ifRead = true; % read data

%% MOVIE PARAMETERS

movieFile = 'moviePhi_cuvocn.avi';

Mov.figWidth   = 1500;    % in pixels
Mov.deltaX     = 40;
Mov.deltaX2    = 50;
Mov.deltaY     = 35;
Mov.deltaY2    = 40;
Mov.gapX       = 50;
Mov.gapY       = 40;
Mov.visible    = 'on';
Mov.fps        = 20;

model = floeVSAModel_den_ose( experiment );

%% READ DATA
if ifRead
    cE = getData( model.embComponent( 1, : ) ); % delay-embedded concentration data 
    uvOcnE = getData( model.embComponent( 2, : ) ); % delay-embedded ocean velocity data
    rhoC = getDensity( model.density( 1 ) ); % density data (concentration)
    rhoUVOcn = getDensity( model.density( 2 ) ); % ocean velocity data

    [ phi, ~, lambda ] = getDiffusionEigenfunctions( model ); 

    nE = size( cE, 1 );      % number of lags in emebdding window
    nG = nX * nY;            % number of gridpoints
    nS = size( cE, 2 ) / nG; % number of samples
    nPhi = size( phi, 2 );   % number of eigenfunctions
    
    cE       = reshape( cE, [ nE nS nY nX ] );
    uvOcnE   = reshape( uvOcnE, [ 2 nE nS nY nX ] ); 
    rhoC     = reshape( rhoC, [ nS nY nX ] );    
    rhoUVOcn = reshape( rhoUVOcn, [ nS nY nX ] );
    phi    = reshape( phi, [ nS nY nX nPhi ] );
end

%% PREPARE MOVIE

nPhiPlt = numel( idxPhi );
Mov.nTileX = 3;
Mov.nTileY = 1 + ceil( nPhiPlt / Mov.nTileX );

Mov.panelX = ( Mov.figWidth - Mov.deltaX - Mov.deltaX2 - ( Mov.nTileX -1 ) * Mov.gapX ) / Mov.nTileX;
Mov.panelY = Mov.panelX / 5;

Mov.posn = [ 0, ...
             0, ...
             Mov.nTileX * Mov.panelX + ( Mov.nTileX - 1 ) * Mov.gapX + Mov.deltaX + Mov.deltaX2, ...
             Mov.nTileY * Mov.panelY + ( Mov.nTileY - 1 ) * Mov.gapY + Mov.deltaY + Mov.deltaY2 ];

writerObj = VideoWriter( movieFile );
writerObj.FrameRate = Mov.fps;
writerObj.Quality = 100;
open( writerObj );

Mov.fig = figure( 'units', 'pixels', ...
          'paperunits', 'points', ...
          'position', Mov.posn, ...
          'paperPosition', Mov.posn, ...
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
          'defaultAxesTickLength', [ 0.005 0 ], ...
          'defaultAxesFontName', 'helvetica', ...
          'defaultTextFontName', 'helvetica', ...
          'defaultAxesLineWidth', 1, ...
          'defaultAxesLayer', 'top' );

Mov.ax = zeros( Mov.nTileX, Mov.nTileY );

for iAx = 1 : Mov.nTileX
    for jAx = 1 : Mov.nTileY
        if sub2ind( [ Mov.nTileY Mov.nTileX ], jAx, iAx ) <= nPhiPlt + Mov.nTileX 
            Mov.ax( iAx, jAx ) = axes( ...
                    'units', 'pixels', ...
                    'position', ...
                    [ Mov.deltaX + ( iAx - 1 ) * ( Mov.panelX + Mov.gapX ), ...
                      Mov.deltaY + ( Mov.nTileY - jAx ) * ( Mov.panelY + Mov.gapY ), ...
                      Mov.panelX, Mov.panelY ] );
        end
    end
end

Mov.axTitle = axes( 'units', 'pixels', 'position', [ Mov.deltaX, Mov.deltaY, ...
                     Mov.nTileX * Mov.panelX + ( Mov.nTileX - 1 ) * Mov.gapX, ...
                     Mov.nTileY * Mov.panelY + ( Mov.nTileY - 1 ) * Mov.gapY + 15 ], ...
                'color', 'none', 'box', 'off' );


% Eulerian coarse grid (concentration)
x = linspace( xLim( 1 ), xLim( 2 ), nX + 1 );
y = linspace( yLim( 1 ), yLim( 2 ), nY + 1 );
pData = zeros( length( y ), length( x ) );

% Eulerian coarse grid (velocity)
xUV = ( x( 1 : end - 1 ) + x( 2 : end ) ) / 2;
yUV = ( y( 1 : end - 1 ) + y( 2 : end ) ) / 2;

%% PLOT MOVIE
for iS = 1 : nS

    disp( sprintf( 'Frame %i/%i', iS, nS ) )

    % concentration
    pData( 1 : end - 1, 1 : end - 1 ) = squeeze( cE( idxE, iS, :, : ) );
    hP = pcolor( Mov.ax( 1, 1 ), x / 1000, y / 1000, pData );
    set( hP, 'edgeColor', 'none' )

    % ocean velocity
    hold( Mov.ax( 1, 1 ), 'on' )
    qData = squeeze( uvOcnE( :, idxE, iS, :, : ) );
    quiver( Mov.ax( 1, 1 ), xUV / 1000, yUV / 1000, ...
            squeeze( qData( 1, :, : ) ), squeeze( qData( 2, :, : ) ), 'r-' ); 
    
    axPos = get( Mov.ax( 1, 1 ), 'position' );
    hC = colorbar( Mov.ax( 1, 1 ), 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 1 ) = cPos( 1 ) + 0.03;
    cPos( 3 ) = cPos( 3 ) * .5;
    set( hC, 'position', cPos );
    set( Mov.ax( 1, 1 ), 'position', axPos, ...
         'cLim', [ 0 1 ], ...
         'xTick', [ xLim( 1 ) : 50 : xLim( 2 ) ], ...
         'yTick', [ yLim( 1 ) : 25 : yLim( 2 ) ], ...
         'xTicklabel', '' ) 
    title( Mov.ax( 1, 1 ), 'Sea ice concentration' )
    set( Mov.ax( 1, 1 ), 'tickDir', 'out' )
    colormap( Mov.ax( 1, 1 ), 'gray' )

    % kernel density (concentration data)
    pData( 1 : end - 1, 1 : end - 1 ) = squeeze( rhoC( iS, :, : ) ); 
    hP = pcolor( Mov.ax( 2, 1 ), x / 1000, y / 1000, pData );
    set( hP, 'edgeColor', 'none' )

    axPos = get( Mov.ax( 2, 1 ), 'position' );
    hC = colorbar( Mov.ax( 2, 1 ), 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 1 ) = cPos( 1 ) + 0.03;
    cPos( 3 ) = cPos( 3 ) * .5;
    set( hC, 'position', cPos );
    set( Mov.ax( 2, 1 ), 'position', axPos, ...
         'cLim', [ 0 max( pData( : ) ) ], ...
         'xTick', [ xLim( 1 ) : 50 : xLim( 2 ) ], ...
         'yTick', [ yLim( 1 ) : 25 : yLim( 2 ) ], ...
         'xTickLabel', '', 'yTickLabel', '' ) 
    title( Mov.ax( 2, 1 ), 'kernel density (concentration)' )
    set( Mov.ax( 2, 1 ), 'tickDir', 'out' )
    colormap( Mov.ax( 2, 1 ), 'gray' )

    % kernel density (ocean velocity data)
    pData( 1 : end - 1, 1 : end - 1 ) = squeeze( rhoUVOcn( iS, :, : ) ); 
    hP = pcolor( Mov.ax( 3, 1 ), x / 1000, y / 1000, pData );
    set( hP, 'edgeColor', 'none' )

    axPos = get( Mov.ax( 3, 1 ), 'position' );
    hC = colorbar( Mov.ax( 3, 1 ), 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 1 ) = cPos( 1 ) + 0.03;
    cPos( 3 ) = cPos( 3 ) * .5;
    set( hC, 'position', cPos );
    set( Mov.ax( 3, 1 ), 'position', axPos, ...
         'cLim', [ 0 max( pData( : ) ) ], ...
         'xTick', [ xLim( 1 ) : 50 : xLim( 2 ) ], ...
         'yTick', [ yLim( 1 ) : 25 : yLim( 2 ) ], ...
         'xTickLabel', '', 'yTickLabel', '' ) 
    title( Mov.ax( 3, 1 ), 'kernel density (ocean velocity)' )
    set( Mov.ax( 3, 1 ), 'tickDir', 'out' )
    colormap( Mov.ax( 3, 1 ), 'gray' )


    % eigenfunctions
    for iPhi = 1 : nPhiPlt
        idx = 3 + iPhi;
        [ iAx, jAx ] = ind2sub( [ Mov.nTileX Mov.nTileY ], idx );

        pData( 1 : end - 1, 1 : end - 1 ) = squeeze( phi( iS, :, :, idxPhi( iPhi ) ) ); 
        hP = pcolor( Mov.ax( iAx, jAx ), x / 1000, y / 1000, pData );
        set( hP, 'edgeColor', 'none' )

        axPos = get( Mov.ax( iAx, jAx ), 'position' );
        hC = colorbar( Mov.ax( iAx, jAx ), 'location', 'eastOutside' );
        cPos = get( hC, 'position' );
        cPos( 1 ) = cPos( 1 ) + 0.03;
        cPos( 3 ) = cPos( 3 ) * .5;
        set( hC, 'position', cPos );
        set( Mov.ax( iAx, jAx ), 'position', axPos, ...
             'cLim', max( abs( pData( : ) ) ) * [ -1 1 ], ...
             'xTick', [ xLim( 1 ) : 50 : xLim( 2 ) ], ...
             'yTick', [ yLim( 1 ) : 25 : yLim( 2 ) ] ) 
        if iAx ~= 1
            set( Mov.ax( iAx, jAx ), 'yTickLabel', '' )
        end
        if jAx ~= Mov.nTileY
            set( Mov.ax( iAx, jAx ), 'xTickLabel', '' )
        end
        title( Mov.ax( iAx, jAx ), sprintf( '\\phi_{%i}, \\lambda_{%i} = %1.3f', ...
               idxPhi( iPhi ) - 1, idxPhi( iPhi ) - 1, ...
               lambda( idxPhi( iPhi ) ) ) )
        set( Mov.ax( iAx, jAx ), 'tickDir', 'out' )
    end

    % time
    t = ( iS - 1 ) * dt / 3600;
    title( Mov.axTitle, sprintf( 'Time = %1.2f hours', t ) )
    axis( Mov.axTitle, 'off' )

    % add frame to movie
    frame = getframe( Mov.fig );
    writeVideo( writerObj, frame )

    % reset axes for next frame
    for iAx = 1 : 3 + nPhiPlt
        cla( Mov.ax( iAx ), 'reset' )
    end
    cla( Mov.axTitle, 'reset' )
end

%% WRAP UP
close( writerObj )
