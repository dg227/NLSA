experiment = { 'ip_sst' };

% MOVIE PARAMETERS
idxC    = 1;        % component index
idxPhi  = { [ 5 27 30 43 44 45 47 50 ] };
nTileX  = 3;
nTileY  = 3;
tagPos  = [ 121 62 ];
panelX  = 150;
panelY  = panelX * 4 / 9;
deltaX  = 30;
deltaX2 = 60; 
deltaY  = 30;
deltaY2 = 25;
gapX    = 10; 45;
gapY    = 5; 10;
fps     = 6;  % frames per second
visible = 'on';


for iModel = 1 : numel( experiment )
    [ model, In ] = ccsmNLSAModel_ssa( experiment{ iModel } );
    nE    = getTrgEmbeddingWindow( model, 1 );
    nD    = getTrgDimension( model, 1 );

    idxAx     = 1 : numel( idxPhi{ iModel } );     % axes indices to plot


    % MOVIE OF A MATRIX
    load( fullfile( getPath( model.srcComponent, idxC ), 'dataGrid.mat' ), 'x', 'y', 'ifXY', 'w' )
    a         = getProjectedData( model, idxC, idxPhi{ iModel } );
    a     = bsxfun( @rdivide, a, repmat( w, [ nE 1 ] ) );
    aMax  = .8 * max( abs( a ), [], 1 );
    %movieDir  = pwd; %getLinearMapPath( model, idxA( iModel ), 'movies' );
    %if ~isdir( movieDir )
    %    mkdir( movieDir )
    %end
    nPhi      = numel( idxPhi{ iModel } );
    movieFile = 'moviePrj_ssa.avi';
    %movieFile = fullfile( movieDir, ...
    %                      [ 'moviePrj', sprintf( '_%i', idxPhi{ iModel } ), '.avi' ] );

   
    xLim = In.Trg( idxC ).xLim;
    yLim = In.Trg( idxC ).yLim;

    posn    = [ 0, ...
                0, ...
                nTileX * panelX + ( nTileX - 1 ) * gapX + deltaX + deltaX2, ...
                nTileY * panelY + ( nTileY - 1 ) * gapY + deltaY + deltaY2 ];

    %writerObj = VideoWriter( fullfile( movieDir, movieFile ) );
    writerObj = VideoWriter( movieFile );
    writerObj.FrameRate = fps;
    writerObj.Quality   = 100;
    open( writerObj );

    fig = figure( 'units', 'pixels', ...
                  'position', posn, ...
                  'visible', visible, ...
                  'color', 'white', ...
                  'doubleBuffer', 'on', ...
                  'backingStore', 'off', ...
                  'defaultAxesTickDir', 'out', ...
                  'defaultAxesNextPlot', 'replace', ...
                  'defaultAxesBox', 'on', ...
                  'defaultAxesFontSize', 6, ...
                  'defaultTextFontSize', 6, ...
                  'defaultAxesTickDir',  'out', ...
                  'defaultAxesTickLength', [ 0.012 0 ], ...
                  'defaultAxesFontName', 'helvetica', ...
                  'defaultTextFontName', 'helvetica', ...
                  'defaultAxesLineWidth', 1, ...
                  'defaultAxesLayer', 'top', ...
                  'renderer', 'zbuffer' );

    ax    = zeros( nTileX, nTileY );
    axPos = cell( nTileX, nTileY );

    iPhi = 1;
    for jAx = 1 : nTileY
        for iAx = 1 : nTileX 
            posnY = deltaY + ( nTileY - jAx ) * ( panelY + gapY );
            axPos{ iAx, jAx } =  [ deltaX + ( iAx - 1 ) * ( panelX + gapX ), posnY, panelX, panelY ]; 
            if iPhi <= nPhi
                 ax( iAx, jAx ) = axes( 'units', 'pixels', ...
                                       'position', [ deltaX + ( iAx - 1 ) * ( panelX + gapX ), ...
                                        posnY, panelX, panelY ] );
            end
            iPhi = iPhi + 1;
        end
    end              

    axTitle = axes( 'units', 'pixels', 'position', [ deltaX, deltaY, ...
                                  nTileX * panelX + ( nTileX - 1 ) * gapX, ...
                                  nTileY * panelY + ( nTileY - 1 ) * gapY ], ...
                    'color', 'none', 'box', 'off' );

    cData            = zeros( size( ifXY ) );
    cData( ~ifXY )   = NaN; 
    cPos             = cell( size( ax ) );



    for iE = 1 : nE
        disp( sprintf( 'Lag %i/%i', iE, nE ) )

        idxD1         = ( iE - 1 ) * nD + 1;
        idxD2         = iE * nD;

        for iPhi = 1 : nPhi
            cData( ifXY ) = a( idxD1 : idxD2, iPhi );
                    
            set( gcf, 'currentAxes', ax( idxAx( iPhi ) ) )
        
            m_proj( 'equidistant cylindrical','lat',[ yLim( 1 ) yLim( 2 ) ],'long',[ xLim( 1 ) xLim( 2 ) ] );
            h = m_pcolor( x, y, cData );
            set( h, 'edgeColor', 'none' )
            [ subPat( 1 ), subPat( 2 ) ] = ind2sub( [ nTileX nTileY ], idxAx( iPhi ) );
            if subPat( 1 ) == 1 && subPat( 2 ) ~= nTileY
                m_grid( 'linest', 'none', ...
                        'linewidth', 2 , ...
                        'tickdir','out', ...
                        'xTick', [ xLim( 1 ) : 40 : xLim( 2 ) ], ...
                        'yTick', [ yLim( 1 ) : 20 : yLim( 2 ) ], ...
                        'xTickLabels', [] ); %, ...
                        %'yTickLabels', [ '20N'; '30N'; '40N'; '50N'; '60N' ] );
            elseif subPat( 1 ) ~= 1 && subPat( 2 ) == nTileY
                m_grid( 'linest', 'none', ...
                        'linewidth', 2 , ...
                        'tickdir','out', ...
                        'xTick', [ xLim( 1 ) : 40 : xLim( 2 ) ], ...
                        'yTick', [ yLim( 1 ) : 20 : yLim( 2 ) ], ...
                        'yTickLabels', [] ); %, ...
                        %'xTickLabels', ['120E'; '140E'; '160E'; '180 '; '160W'; '140W'; '120W' ], ...
                        %'yTickLabels', [] );
            elseif subPat( 1 ) == 1 && subPat( 2 ) == nTileY
                m_grid( 'linest', 'none', ...
                        'linewidth', 2 , ...
                        'tickdir','out', ...
                        'xTick', [ xLim( 1 ) : 40 : xLim( 2 ) ], ...
                        'yTick', [ yLim( 1 ) : 20 : yLim( 2 ) ] ); %, ...
                        %'xTick', [ 120 : 20 : 240 ], ...
                        %'yTick', [ 20 : 10 : 60 ], ...
                        %'xTickLabels', ['120E'; '140E'; '160E'; '180 '; '160W'; '140W'; '120W' ], ...
                        %'yTickLabels', [ '20N'; '30N'; '40N'; '50N'; '60N' ] );
            else
                m_grid( 'linest', 'none', ...
                        'linewidth', 2 , ...
                        'tickdir','out', ...
                        'xTick', [ xLim( 1 ) : 40 : xLim( 2 ) ], ...
                        'yTick', [ yLim( 1 ) : 20 : yLim( 2 ) ],  ...
                        'xTickLabels', [], ...
                        'yTickLabels', [] );
            end     
            m_coast( 'line', 'linewidth', 2, 'color', 'k' );
            m_coast( 'line', 'linewidth', 2, 'color', 'k' );
            [ X, Y ] = m_ll2xy( tagPos( 1 ), tagPos( 2 ) );
            title(sprintf( 'phi%i', idxPhi{ iModel }( iPhi ) ) )
            %hC     = colorbar( 'peer', ax( idxAx( iPhi ) ), 'eastoutside' );
            hC = colorbar;
            if iE == 1
                cPos{ idxAx( iPhi ) }   = get( hC, 'position' );
                cPos{ idxAx( iPhi ) }( 1 ) = cPos{ idxAx( iPhi ) }( 1 ) + 0.03;
                cPos{ idxAx( iPhi ) }( 3 ) = cPos{ idxAx( iPhi ) }( 3 ) * .3;
            end
            set( gca, 'cLim', [ -1 1 ] * aMax( iPhi ), 'position', axPos{ idxAx( iPhi ) } )
            set( hC, 'position', cPos{ idxAx( iPhi ) } )
                
        end
	
        title( axTitle,  sprintf( 'Lag %i',  iE ) )
        axis( axTitle, 'off' )

        set( fig, 'position', posn )
        drawnow
        img = print( '-RGBImage' );
        writeVideo( writerObj, im2frame( img ) );
        for iAx = 1 : numel( ax )
           cla( ax( iAx ), 'reset' )
        end
        cla( axTitle, 'reset' )

    end

    close( writerObj );
    close( fig );
end



