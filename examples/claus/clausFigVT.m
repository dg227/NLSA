experiment = 'tb_central_at';
model = clausNLSAModel( experiment );

%% TEMPORAL PATTERN PARAMETERS
idxA      = 1;       % linear map index
idxV      = 1 : 7;    % temporal patterns to plot
figWidth  = 6;      % in inches
deltaX    = 0.35;
deltaX2   = 0.4;
deltaY    = 0.45;
deltaY2   = 0.05;
gapX      = 0.2;
gapY      = 0.1;
tPlotLim  = { '1992010100' '1993123121' };     % time limits 
tFormat = 'yyyymmddhh';
vPlotLim  = [ -3 3 ];

            
%% TEMPORAL PATTERN FIGURES
vT      = getTemporalPatterns( model, idxA );
tNum    = getSrcTime( model );
idxT1   = getOrigin( model.embComponent );
figDir  = fullfile( getLinearMapPath( model, idxA ), 'figures' );
figFile = [ 'figVT',   sprintf( '_%i', idxV ), '.eps' ];
figFile = fullfile( figDir, figFile );


% Determine dataset indices to plot
iPlotLim = [ find( tNum == datenum( tPlotLim{ 1 }, tFormat ) ) ...
             find( tNum == datenum( tPlotLim{ 2 }, tFormat ) ) ];
iPlotLimNLSA = iPlotLim - idxT1 + 1;
tPlot = datestr( tNum( iPlotLim( 1 ) : iPlotLim( end ) ), 'yymm' );
nPlot = size( tPlot, 1 );
iTick = 1 : 720 : nPlot; % tick every two months

nTileX = 2;
nTileY = numel( idxV );
panelX = ( figWidth - deltaX - deltaX2 - ( nTileX -1 ) * gapX ) / nTileX; 
panelY = panelX * ( 3 / 4 ) ^2;
        
posn     = [ 0, ...
             0, ...
             nTileX * panelX + ( nTileX - 1 ) * gapX + deltaX + deltaX2, ...
             nTileY * panelY + ( nTileY - 1 ) * gapY + deltaY + deltaY2 ];

fig = figure( 'units', 'inches', ...
              'paperPosition', posn, ...
              'position', posn, ...
              'defaultAxesNextPlot', 'add', ...
              'defaultAxesBox', 'on', ...
              'defaultAxesFontSize', 7, ...
              'defaultTextFontSize', 8, ...
              'defaultAxesTickDir', 'out', ...
              'defaultAxesTickLength', [ 0.02 0 ], ...
              'defaultAxesFontName', 'times', ...
              'defaultTextFontName', 'times', ...
              'defaultAxesLayer', 'top' );

ax = zeros( nTileX, nTileY );

for iAx = 1 : nTileX
    for jAx = 1 : nTileY
        ax( iAx, jAx ) = axes( 'units', 'inches', ...
                               'position', [ deltaX + ( iAx - 1 ) * ( panelX + gapX ), ...
                                deltaY + ( nTileY - jAx ) * ( panelY + gapY ), ...
                                panelX, panelY ] );
    end
end

    
nT        = size( vT, 1 ); % number of samples
T         = ( nT - 1 ) / 8 ;    % length of time series in days
dNu       = 1 / T; % sampling interval in frequency space
nNu       = floor( nT / 2 ) + 1;
nuVals    = ( 0 : nNu - 1 ) * dNu; 
    

for iV = 1 : numel( idxV )
        
    vFourier = fft( vT( :, idxV( iV ) ) );
    vPower   = abs( vFourier( 1 : nNu ) ) .^ 2;
        
    axes( ax( 1, iV ) )
    plot( 1 : nPlot, vT( iPlotLimNLSA( 1 ) : iPlotLimNLSA( 2 ), idxV( iV ) ), 'k-' )
    set( gca, 'yLim', vPlotLim, 'xLim', [ 1 nPlot ], 'xTick', iTick )
    if iV == nTileY
        set( gca, 'xTickLabel', tPlot( iTick, 1 : 4 ) )
        xlabel( 'date (yymm)', 'fontSize', 8 )
    else
        set( gca, 'xTickLabel', [] )
    end
    hL = ylabel( [ '{\itv}_{', int2str( idxV( iV ) ), '}' ] );
    hLP = get( hL, 'position' );
    hLP( 1 ) = hLP( 1 ) * 0.8;
    set( hL, 'position', hLP )
    MidTick( 'XY' );
        
    axes( ax( 2, iV ) )
    plot( nuVals, vPower, 'k-' )
    set( gca, 'xScale', 'log', 'xLim', [ 1E-3 1E1 ], 'xTick', [ 1E-4 1E-3, 1E-2 1E-1 1E0 1E1 ], ...
        'xTickLabel', { '1E-4', '1E-3', '1E-2', '1E-1', '1E0', '1E1' }, 'yAxis', 'right', ...
        'yScale', 'log', 'yLim', [ 1E2, 1E9 ], 'yTick', [ 1E2 1E3 1E4, 1E5, 1E6, 1E7, 1E8 1E9 1E10 ], ...
        'yTickLabel', { '1E2', '1E3', '1E4', '1E5', '1E6', '1E7', '1E8', '1E9', '1E10' } )
    if iV == nTileY
        xlabel( 'frequency (d^{-1})', 'fontSize', 8 )
    else
        set( gca, 'xTickLabel', [] )
    end
    hL = ylabel( [ '|FT({\itv}_{', int2str( idxV( iV ) ), '})|^2' ], 'fontSize', 7 );
    %hLP = get( hL, 'position' );
    %hLP( 1 ) = hLP( 1 ) * .75;
    %set( hL, 'position', hLP )
%    text( tagPos( 1 ), tagPos( 2 ), tagStr{ iV } )
    
end

print( '-deps2', '-r1200', figFile )
