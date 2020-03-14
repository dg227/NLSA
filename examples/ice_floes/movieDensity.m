%% PLOT MOVIE OF KERNEL DENSITY ESTIMATE

%% DATASET/NLSA PARAMETERS
experiment = 'channel'; 

xLim =  [ -2.5E5 2.5E5 ]; 
yLim =  [ -5E4 5E4 ]; 
nX = 25;  % number of coarse cells, x direction
nY = 5;   % number of coarse cells, y direction

dt = 20;  % time step (sec)
nDT = 50; % number of timesteps between output

idxE = 1; % index within embedding window to plot raw data.

%% SCRIPT OPTIONS
ifRead = true; % read data

%% MOVIE PARAMETERS

movieFile = 'movieRho.avi';

Mov.figWidth   = 500;    % in pixels
Mov.deltaX     = 40;
Mov.deltaX2    = 50;
Mov.deltaY     = 35;
Mov.deltaY2    = 40;
Mov.gapX       = 30;
Mov.gapY       = 40;
Mov.visible    = 'on';
Mov.fps        = 20;

model = floeVSAModel_den_ose( 'channel' );

%% READ DATA
if ifRead
    cE = getData( model.embComponent( 1, : ) ); % delay-embedded concentration data 
    rho = getDensity( model ); % density data

    nE = size( cE, 1 );      % number of lags in emebdding window
    nG = nX * nY;            % number of gridpoints
    nS = size( cE, 2 ) / nG; % number of samples
    
    cE = reshape( cE, [ nE nS nY nX ] );
    rho = reshape( rho, [ nS nY nX ] );    
end

%% PREPARE MOVIE
Mov.nTileX = 1;
Mov.nTileY = 2;

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
        Mov.ax( iAx, jAx ) = axes( ...
                'units', 'pixels', ...
                'position', ...
                [ Mov.deltaX + ( iAx - 1 ) * ( Mov.panelX + Mov.gapX ), ...
                  Mov.deltaY + ( Mov.nTileY - jAx ) * ( Mov.panelY + Mov.gapY ), ...
                  Mov.panelX, Mov.panelY ] );
    end
end

Mov.axTitle = axes( 'units', 'pixels', 'position', [ Mov.deltaX, Mov.deltaY, ...
                     Mov.nTileX * Mov.panelX + ( Mov.nTileX - 1 ) * Mov.gapX, ...
                     Mov.nTileY * Mov.panelY + ( Mov.nTileY - 1 ) * Mov.gapY + 15 ], ...
                'color', 'none', 'box', 'off' );


% Eulerian coarse grid
x = linspace( xLim( 1 ), xLim( 2 ), nX + 1 );
y = linspace( yLim( 1 ), yLim( 2 ), nY + 1 );
cPlt = zeros( length( y ), length( x ) );
rhoPlt = cPlt; 

% Colormap

%% PLOT MOVIE
for iS = 1 : nS

    disp( sprintf( 'Frame %i/%i', iS, nS ) )

    % concentration
    set( gcf, 'currentAxes', Mov.ax( 1, 1 ) )
    cPlt( 1 : end - 1, 1 : end - 1 ) = squeeze( cE( idxE, iS, :, : ) );
    hP = pcolor( Mov.ax( 1, 1 ), x / 1000, y / 1000, cPlt );
    set( hP, 'edgeColor', 'none' )
    
    axPos = get( Mov.ax( 1, 1 ), 'position' );
    hC = colorbar( Mov.ax( 1, 1 ), 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 1 ) = cPos( 1 ) + 0.1;
    cPos( 3 ) = cPos( 3 ) * .5;
    set( hC, 'position', cPos );
    set( Mov.ax( 1, 1 ), 'position', axPos, ...
         'cLim', [ 0 1 ], ...
         'xTick', [ xLim( 1 ) : 50 : xLim( 2 ) ], ...
         'yTick', [ yLim( 1 ) : 25 : yLim( 2 ) ] ) 
    title( Mov.ax( 1, 1 ), 'Sea ice concentration' )
    set( Mov.ax( 1, 1 ), 'tickDir', 'out' )

    % kernel density
    set( gcf, 'currentAxes', Mov.ax( 1, 2 ) )
    rhoPlt( 1 : end - 1, 1 : end - 1 ) = squeeze( rho( iS, :, : ) ); 
    hP = pcolor( Mov.ax( 1, 2 ), x / 1000, y / 1000, rhoPlt );
    set( hP, 'edgeColor', 'none' )

    axPos = get( Mov.ax( 1, 2 ), 'position' );
    hC = colorbar( Mov.ax( 1, 2 ), 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 1 ) = cPos( 1 ) + 0.1;
    cPos( 3 ) = cPos( 3 ) * .5;
    set( hC, 'position', cPos );
    set( Mov.ax( 1, 2 ), 'position', axPos, ...
         'cLim', [ 0 max( rhoPlt( : ) ) ], ...
         'xTick', [ xLim( 1 ) : 50 : xLim( 2 ) ], ...
         'yTick', [ yLim( 1 ) : 25 : yLim( 2 ) ] ) 
    title( Mov.ax( 1, 2 ), 'kernel density' )
    set( Mov.ax( 1, 2 ), 'tickDir', 'out' )

    % time
    t = ( iS - 1 ) * dt / 3600;
    title( Mov.axTitle, sprintf( 'Time = %1.2f hours', t ) )
    axis( Mov.axTitle, 'off' )

    colormap( Mov.ax( 1, 1 ), 'gray' )
    colormap( Mov.ax( 1, 2 ), 'gray' )

    % add frame to movie
    frame = getframe( Mov.fig );
    writeVideo( writerObj, frame )

    % reset axes for next frame
    cla( Mov.ax( 1, 1 ), 'reset' )
    cla( Mov.ax( 1, 2 ), 'reset' )
    cla( Mov.axTitle, 'reset' )

end

%% WRAP UP
close( writerObj )
