experiment = { 'ip_sst' 'ip_sst' };
idxPhi = { [ 61 : 70 ]  [ 71 : 80 ] }; 
figName = { 'figure7' 'figure8' };
tagStr = { '(a)' '(b)' '(c)' '(d)' '(e)' '(f)' '(g)' '(h)' '(i)' '(j)' '(k)' };
tagPos = [ 4 1E6 ];

figWidth   = 5.3;       % in inches
deltaX     = 0.35;
deltaX2    = 0.5;
deltaY     = 0.45;
deltaY2    = 0.05;
gapX       = 0.3;
gapY       = 0.1;
tPlotLim   = { '020001' '030001' }; 
tFormat    = 'yyyymm';
nTick      = 120; % tick every 10 years
vPlotLim   = [ -3 3 ];
figVisible = 'off';

nTileX = 2;
panelX = ( figWidth - deltaX - deltaX2 - ( nTileX -1 ) * gapX ) / nTileX; 
panelY = panelX * ( 3 / 4 ) ^ 3;
        


for iModel = 1 : numel( experiment )

    model = ccsmNLSAModel_ssa( experiment{ iModel } );
    phi   = getCovarianceEigenfunctions( model );
    phi   = bsxfun( @times, phi, sqrt( size( phi, 1 ) ) );
    tNum  = getSrcTime( model );
    idxT1 = getOrigin( model.embComponent );

    % Determine dataset indices to plot
    iPlotLim = [ find( tNum == datenum( tPlotLim{ 1 }, tFormat ) ) ...
                 find( tNum == datenum( tPlotLim{ 2 }, tFormat ) ) ];
    iPlotLimNLSA = iPlotLim - idxT1 + 1;
    tPlot = datestr( tNum( iPlotLim( 1 ) : iPlotLim( end ) ), 'yyyymm' );
    nPlot = size( tPlot, 1 );
    iTick = 1 : nTick : nPlot; % tick every two years


    nTileY = numel( idxPhi{ iModel } );
    posn     = [ 0, ...
                 0, ...
                 nTileX * panelX + ( nTileX - 1 ) * gapX + deltaX + deltaX2, ...
                 nTileY * panelY + ( nTileY - 1 ) * gapY + deltaY + deltaY2 ];

    fig = figure( 'visible', figVisible, ...
                  'units', 'inches', ...
                  'paperPosition', posn, ...
                  'position', posn, ...
                  'defaultAxesNextPlot', 'add', ...
                  'defaultAxesBox', 'on', ...
                  'defaultAxesFontSize', 7, ...
                  'defaultTextFontSize', 8, ...
                  'defaultAxesTickDir', 'out', ...
                  'defaultAxesTickLength', [ 0.02 0 ], ...
                  'defaultAxesFontName', 'LucidaGrande', ...
                  'defaultTextFontName', 'LucidaGrande', ...
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

    
    nT        = size( phi, 1 ); % number of samples
    T         = ( nT - 1 ) /12;    % length of time series in years
    dNu       = 1 / T; % sampling interval in frequency space
    nNu       = floor( nT / 2 ) + 1;
    nuVals    = ( 0 : nNu - 1 ) * dNu; 
    

    for iPhi = 1 : numel( idxPhi{ iModel } )
        
        vFourier = fft( phi( :, idxPhi{ iModel }( iPhi ) ) );
        vPower   = abs( vFourier( 1 : nNu ) ) .^ 2;
        
        axes( ax( 1, iPhi ) )
        plot( 1 : nPlot, phi( iPlotLimNLSA( 1 ) : iPlotLimNLSA( 2 ), idxPhi{ iModel }( iPhi ) ), 'k-' )
        set( gca, 'yLim', vPlotLim, 'xLim', [ 1 nPlot ], 'xTick', iTick )
        if iPhi == nTileY
            set( gca, 'xTickLabel', tPlot( iTick, 2 : 4 ) )
            xlabel( 'date (y)', 'fontSize', 8 )
        else
            set( gca, 'xTickLabel', [] )
        end
        hL = ylabel( [ '{\phi}_{', int2str( idxPhi{ iModel }( iPhi ) ), '}' ] );
        hLP = get( hL, 'position' );
        hLP( 1 ) = hLP( 1 ) * 0.8;
        set( hL, 'position', hLP )
        %MidTick( 'XY' );
        
        axes( ax( 2, iPhi ) )
        plot( nuVals, vPower, 'k-' )
        set( gca, 'xScale', 'log', 'xLim', [ 1E-3 1E1 ], 'xTick', [ 1E-4 1E-3, 1E-2 1E-1 1E0 1E1 ], ...
            'xTickLabel', { '1E-4', '1E-3', '1E-2', '1E-1', '1E0', '1E1' }, 'yAxis', 'right', ...
            'yScale', 'log', 'yLim', [ 1E-2, 1E7 ], 'yTick', [ 1E-2 1E-1 1E0 1E1 1E2 1E3 1E4, 1E5, 1E6, 1E7, 1E8 1E9 1E10 ], ...
            'yTickLabel', { '1E-2', '1E-1', '1E0', '1E1', '1E2', '1E3', '1E4', '1E5', '1E6', '1E7', '1E8', '1E9', '1E10' } )
        if iPhi == nTileY
            xlabel( 'frequency (1/y)', 'fontSize', 8, 'fontName', 'LucidaGrande' )
        else
            set( gca, 'xTickLabel', [] )
        end
        hL = ylabel( [ '|FT({\phi}_{', int2str( idxPhi{ iModel }( iPhi ) ), '})|^2' ], 'fontSize', 7, 'fontName', 'LucidaGrande'  );
        %hLP = get( hL, 'position' );
        %hLP( 1 ) = hLP( 1 ) * .75;
        %set( hL, 'position', hLP )
        text( tagPos( 1 ), tagPos( 2 ), tagStr{ iPhi } )
    
    end
    print( '-deps', '-r300', figName{ iModel } )
    %printeps( gcf, figName{ iModel }, 1200 )
end

