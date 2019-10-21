% KERNEL ANALOG FORECASTING OF NOAA DATASET
%
% Modified 2019/06/21

%% MAIN CALCULATION PARAMETERS AND OPTIONS
experiment     = 'ip_sst';     % forecast model 

idxPhi      = 1 : 75;    % hypothesis space basis functions 
idxC        = 1;          % data component to predict
idxR        = 1;          % realization to predict
nT          = 24;         % prediction timesteps (max. lead time)
thresh      = 0.5;        % anomaly thershold (for characteristic function)
tNum0Plt    = '199001';   % initialization time to plot forecast
tNum0Plt    = '201612';   % initialization time to plot forecast
idxTPltPred = [ 0 : 23 ] + 1; % lead times to plot
idxTPltRunning = [ 0 : 3 : 12 ] + 1; % lead times for running forecast


%% SCRIPT EXCECUTION OPTIONS
ifRead        = true;  % read data and kernel eigenfunctions 
ifPred        = true;  % prediction 
ifErr         = true;  % prediction error 
ifVar         = true;  % forecast variance
ifProb        = true;  % conditional probability prediction
ifProbErr     = true;  % conditional probability error

ifPlotPred    = true;    % prediction and skill metric plots
ifPlotProb    = true;     % conditional probability plot
ifPlotRunning = true;    % running prediction plot
ifPlotProbRunning = true; % running condition probability plot

%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES
[ model, Pars, ParsO ] = noaaNLSAModel_den_ose( experiment );
nX   = getDimension( model.trgComponent( idxC ) ); % dimension of prediction observable
nPhi = numel( idxPhi ); % number of basis functions

%% READ DATA FROM MODEL
if ifRead
    tic
    disp( 'Data retrieval')

    % eigenfunctions read from both forecast and reference models
    % phiO and phiORef are out-of-sample eigenfunctions
    [ phi, mu, lambda ] = getDiffusionEigenfunctions( model ); 
    phiO = getOseDiffusionEigenfunctions( model );

    nS = numel( mu );
    nSO = size( phiO ,1 );

    % training signal 
    f     = getData( model.trgComponent( idxC, idxR ) );
    f     = f( :, getOrigin( model.trgEmbComponent( idxC, idxR ) ) : end );
    fMean = mean( f, 2 );   
    fStd  = std( f, 0, 2 );

    % verification signal
    tStr = [ ParsO.Res( idxR ).tLim{ 1 } '-' ParsO.Res( idxR ).tLim{ 2 } ];

    xyStr = sprintf( 'x%i-%i_y%i-%i', ParsO.Trg( idxC ).xLim( 1 ), ...
                                      ParsO.Trg( idxC ).xLim( 2 ), ...
                                      ParsO.Trg( idxC ).yLim( 1 ), ...
                                      ParsO.Trg( idxC ).yLim( 2 ) );

    pathOut = fullfile( pwd, ...
                        'data/raw',  ...
                        ParsO.Res( idxR ).experiment, ...
                        ParsO.Trg( idxC ).field,  ...
                        [ xyStr '_' tStr ] );

        
    FOut = load( fullfile( pathOut, 'dataX.mat' ), 'x' );
    fOut = FOut.x( :, getOrigin( model.outEmbComponent( idxC, idxR ) ) : end );
    
    % training and verification signals for characteristic function (conditional probability)
    chi     = bsxfun( @minus, f, thresh ) > 0;
    chiMean = mean( chi, 2 );
    chiStd   = std( chi, 0, 2 ); 
    chiOut  = bsxfun( @minus, fOut, thresh ) > 0;
    
    % serial time numbers for verification dataset
    tNumOut = model.outTime{ idxR }( getOrigin( model.outEmbComponent( idxC, idxR ) ) : end ); 
    toc
end


if ifPred
    tic
    disp( 'Prediction' )
    % create time-shifted copies of the input signal
    fTau = lembed( f, [ 1 nS ] + nT - 1, 1 : nT );

    % compute regression coefficients of the time-shifted signal against 
    % eigenfunctions
    cTau = fTau * bsxfun( @times, phi( :, idxPhi ), mu ); 

    % evaluate predictions using out-of-sample eigenfunctions
    fPred = cTau * phiO( :, idxPhi )';
    toc
end

if ifErr
    tic
    disp( 'Prediction error' )
    % create true signal by time-shifting out-of-sample data
    fTrue = lembed( fOut, [ 1 nSO ] + nT - 1, 1 : nT );

    % compute normalized RMSE
    %predErr = fPred - fTrue;
    %fRmse = vecnorm( predErr, 2, 2 ) / sqrt( nSO );
    fRmse = rmse( fPred, fTrue );
    fRmse = bsxfun( @rdivide, reshape( fRmse, [ nX nT ] ), fStd ); 

    % compute pattern correlation score
    fPC = pc ( fPred, fTrue );
    fPC = reshape( fPC, [ nX nT ] );
    toc
end

if ifVar
    tic
    disp( 'Forecast variance' )
    
    % in-sample error
    fErr = ( fTau - cTau * phi( :, idxPhi )' ) .^ 2;

    % compute regression coefficients of in-sample error against 
    % eigenfunctions
    c2Tau = fErr * bsxfun( @times, phi( :, idxPhi ), mu ); 
     
    % evaluate error forecast
    fErrPred = abs( c2Tau * phiO( :, idxPhi )' );
    toc

    % compute predicted RMSE
    fRmsePred = sqrt( vecnorm( fErrPred, 1, 2 ) / nSO );
    fRmsePred = bsxfun( @rdivide, reshape( fRmsePred, [ nX nT ] ), fStd );
end


if ifProb
    tic
    disp( 'Conditional probability forecast' )

    % create time-shifted copies of the characteristic function, after 
    % removal of initial portion due to delay embedding 
    chiTau = lembed( chi, [ 1 nS ] + nT - 1, 1 : nT );

    % compute regression coefficients of the time-shifted signal against 
    % eigenfunctions
    cChiTau = chiTau * bsxfun( @times, phi( :, idxPhi ), mu ); 

    % evaluate predictions using out-of-sample eigenfunctions
    chiPred = cChiTau * phiO( :, idxPhi )';
    chiPred( chiPred > 1 ) = 1;
    chiPred( chiPred < 0 ) = 0;
    toc
end

if ifProbErr
    tic
    disp( 'Conditional probability error' )
    % create true signal by time-shifting out-of-sample data
    chiTrue = lembed( chiOut, [ 1 nSO ] + nT - 1, 1 : nT );

    % compute normalized RMSE
    %chiPredErr = chiPred - chiTrue;
    %chiRmse = vecnorm( chiPredErr, 2, 2 ) / sqrt( nSO );
    chiRmse = rmse( chiPred, chiTrue );
    chiRmse = bsxfun( @rdivide, reshape( chiRmse, [ nX nT ] ), chiStd ); 

    % compute pattern correlation score
    chiPC = pc( chiPred, chiTrue );
    toc
end
    
if ifPlotPred

    % Set up figure size
    Fig.figWidth  = 4;    % in inches
    Fig.deltaX    = .5;
    Fig.deltaX2   = .15;
    Fig.deltaY    = .45;
    Fig.deltaY2   = .25;
    Fig.gapX      = .35;
    Fig.gapY      = .5;


    nTileX = nX;
    nTileY = 3;

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 - ( nTileX - 1 ) * Fig.gapX ) / nTileX;
    panelY = panelX * 3 / 4;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Fig.gapX + Fig.deltaX + Fig.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Fig.gapY + Fig.deltaY + Fig.deltaY2 ];


    fig = figure( 'units', 'inches', ...
                  'paperunits', 'inches', ...
                  'position', posn, ...
                  'paperPosition', posn, ...
                  'color', 'white', ...
                  'doubleBuffer', 'on', ...
                  'backingStore', 'off', ...
                  'defaultAxesTickDir', 'out', ...
                  'defaultAxesNextPlot', 'replace', ...
                  'defaultAxesBox', 'on', ...
                  'defaultAxesFontSize', 8, ...
                  'defaultTextFontSize', 8, ...
                  'defaultAxesTickDir',  'out', ...
                  'defaultAxesTickLength', [ 0.02 0 ], ...
                  'defaultAxesFontName', 'helvetica', ...
                  'defaultTextFontName', 'helvetica', ...
                  'defaultAxesLayer', 'top' );

    ax = zeros( nTileX, nTileY );

    for iAx = 1 : nTileX
        for jAx = 1 : nTileY
            ax( iAx, jAx ) = axes( ...
                    'units', 'inches', ...
                    'position', ...
                    [ Fig.deltaX + ( iAx - 1 ) * ( panelX + Fig.gapX ), ...
                      Fig.deltaY + ( nTileY - jAx ) * ( panelY + Fig.gapY ), ...
                      panelX, panelY ] );
        end
    end


    % find initialization time index and construct tick labels for time axis
    idxT0Plt = find( datenum( tNum0Plt, 'yyyymm' ) == tNumOut, 1, 'first' );
    tNumPlt = tNumOut( idxT0Plt + idxTPltPred - 1 ); 
    tLabels = cellstr( datestr( tNumPlt, 'mm/yy' ) ); 
    tVals = [ 0 : nT - 1 ];

    for iX = 1 : nX

        jdxTPltPred = iX + ( 0 : nT - 1 ) * nX;

        set( gcf, 'currentAxes', ax( iX, 1 ) )
        plot( tNumPlt, fTrue( jdxTPltPred, idxT0Plt ), 'k-' )    
        hold on
        plot( tNumPlt, fPred( jdxTPltPred, idxT0Plt ), 'b-' )
        plot( tNumPlt, fPred( jdxTPltPred, idxT0Plt ) ...
                    + sqrt( fErrPred( jdxTPltPred, idxT0Plt ) ), 'c-' )
        plot( tNumPlt, fPred( jdxTPltPred, idxT0Plt ) ...
                    - sqrt( fErrPred( jdxTPltPred, idxT0Plt ) ), 'c-' )
        if iX == 1
            hL = legend( 'true', 'forecast', 'estimated error', 'location', 'southWest' );
            %hL = legend( 'true', 'forecast', 'location', 'northEast' );
            %lPos = get( hL, 'position' ); 
            %lPos( 2 ) = lPos( 2 ) - .7;
            %set( hL, 'position', lPos )
        end
        %xlabel( '\tau' )
        title( 'Nino 3.4 index' )
        set( gca, 'xLimSpec', 'tight', 'xTick', tNumPlt( 1 : 3 : end ), ...
            'xTickLabel', tLabels( 1 : 3 : end )  )
        if iX == 1
            ylabel( 'SST anomaly (K)' )
        end


        set( gcf, 'currentAxes', ax( iX, 2 ) )
        plot( tVals, fRmse( iX, idxTPltPred ), 'b-' )
        hold on 
        plot( tVals, fRmsePred( iX, idxTPltPred ), 'c-' ) 
        set( gca, 'xLimSpec', 'tight' )
        set( gca, 'yLim', [ 0 1.2 ], 'yTick', [ 0 : .2 : 1.2 ] )
        grid on
        xlabel( '\tau' )
        if iX == 1
            ylabel( 'normalized L2 error' )
        end
        legend( 'true error', 'estimated error', 'location', 'southWest' )

        set( gcf, 'currentAxes', ax( iX, 3 ) )
        plot( tVals, fPC( iX, idxTPltPred ), 'b-' )
        set( gca, 'xLimSpec', 'tight' )
        set( gca, 'yLim', [ .3 1 ], 'yTick', [ .3 : .1 : 1 ] )
        grid on
        xlabel( 'lead time \tau (months)' )
        if iX == 1
            ylabel( 'pattern correlation' )
        end

        % bring legend to foreground
        aax = get( gcf, 'children' );
        ind = find( isgraphics( aax, 'Legend' ) );
        set( gcf, 'children', aax( [ ind : end, 1 : ind - 1 ] ) )
        %legend boxoff
    end

    print -dpng -r300 figNino34Pred.png
end



if ifPlotProb

    % Set up figure size
    Fig.figWidth  = 4;    % in inches
    Fig.deltaX    = .5;
    Fig.deltaX2   = .15;
    Fig.deltaY    = .45;
    Fig.deltaY2   = .25;
    Fig.gapX      = .35;
    Fig.gapY      = .5;


    nTileX = nX;
    nTileY = 2;

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 - ( nTileX -1 ) * Fig.gapX ) / nTileX;
    panelY = panelX * 3 / 4;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Fig.gapX + Fig.deltaX + Fig.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Fig.gapY + Fig.deltaY + Fig.deltaY2 ];


    fig = figure( 'units', 'inches', ...
                  'paperunits', 'inches', ...
                  'position', posn, ...
                  'paperPosition', posn, ...
                  'color', 'white', ...
                  'doubleBuffer', 'on', ...
                  'backingStore', 'off', ...
                  'defaultAxesTickDir', 'out', ...
                  'defaultAxesNextPlot', 'replace', ...
                  'defaultAxesBox', 'on', ...
                  'defaultAxesFontSize', 8, ...
                  'defaultTextFontSize', 8, ...
                  'defaultAxesTickDir',  'out', ...
                  'defaultAxesTickLength', [ 0.02 0 ], ...
                  'defaultAxesFontName', 'helvetica', ...
                  'defaultTextFontName', 'helvetica', ...
                  'defaultAxesLayer', 'top' );

    ax = zeros( nTileX, nTileY );

    for iAx = 1 : nTileX
        for jAx = 1 : nTileY
            ax( iAx, jAx ) = axes( ...
                    'units', 'inches', ...
                    'position', ...
                    [ Fig.deltaX + ( iAx - 1 ) * ( panelX + Fig.gapX ), ...
                      Fig.deltaY + ( nTileY - jAx ) * ( panelY + Fig.gapY ), ...
                      panelX, panelY ] );
        end
    end

    % find initialization time index and construct tick labels for time axis
    idxT0Plt = find( datenum( tNum0Plt, 'yyyymm' ) == tNumOut, 1, 'first' );
    tNumPlt = tNumOut( idxT0Plt + idxTPltPred - 1 ); 
    tLabels = cellstr( datestr( tNumPlt, 'mm/yy' ) ); 
    tVals = ( idxTPltPred - 1 );
    for iX = 1 : nX

        jdxTPltPred = iX + ( 0 : nT - 1 ) * nX;

        set( gcf, 'currentAxes', ax( iX, 1 ) )
        plot( tNumPlt, chiTrue( jdxTPltPred, idxT0Plt ), 'k-' )    
        hold on
        plot( tNumPlt, chiPred( jdxTPltPred, idxT0Plt ), 'b-' )
        if iX == 1
            hL = legend( 'true', 'forecast', 'location', 'northEast' );
            lPos = get( hL, 'position' ); 
            lPos( 2 ) = lPos( 2 ) - .7;
            set( hL, 'position', lPos )
        end
        xlabel( '\tau' )
        if iX == 1
            ylabel( 'probability' )
        end
        title( sprintf( 'Conditional probability; Threshold = %1.1f', thresh ) )
        set( gca, 'xLimSpec', 'tight', 'xTick', tNumPlt( 1 : 3 : end ), ...
            'xTickLabel', tLabels( 1 : 3 : end ), 'yLim', [ -.1 1.1 ], 'yTick', [ 0 : .25 : 1 ] )
        grid on

        set( gcf, 'currentAxes', ax( iX, 2 ) )
        plot( tVals, chiRmse( iX, idxTPltPred ), 'b-' )
        hold on 
        set( gca, 'xLimSpec', 'tight' )
        set( gca, 'yLim', [ 0 2 ], 'yTick', [ 0 : .2 : 1.4 ] )
        grid on
        xlabel( 'lead time \tau (months)' )
        if iX == 1
            ylabel( 'normalized L2 error' )
        end
        %legend( 'true error', 'estimated error', 'location', 'southeast' )

        % bring legend to foreground
        aax = get( gcf, 'children' );
        ind = find( isgraphics( aax, 'Legend' ) );
        set( gcf, 'children', aax( [ ind : end, 1 : ind - 1 ] ) )
        %legend boxoff
    end

    print -dpng -r300 figNino34Prob.png
end


if ifPlotRunning
    
    nTPlt = numel( idxTPltRunning );

    % Set up figure size
    Fig.figWidth  = 6;    % in inches
    Fig.deltaX    = .5;
    Fig.deltaX2   = .15;
    Fig.deltaY    = .45;
    Fig.deltaY2   = .25;
    Fig.gapX      = .35;
    Fig.gapY      = .5;


    nTileX = nX;
    nTileY = nTPlt;

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 - ( nTileX -1 ) * Fig.gapX ) / nTileX;
    panelY = panelX * ( 3 / 4 ) ^ 3;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Fig.gapX + Fig.deltaX + Fig.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Fig.gapY + Fig.deltaY + Fig.deltaY2 ];


    fig = figure( 'units', 'inches', ...
                  'paperunits', 'inches', ...
                  'position', posn, ...
                  'paperPosition', posn, ...
                  'color', 'white', ...
                  'doubleBuffer', 'on', ...
                  'backingStore', 'off', ...
                  'defaultAxesTickDir', 'out', ...
                  'defaultAxesNextPlot', 'replace', ...
                  'defaultAxesBox', 'on', ...
                  'defaultAxesFontSize', 8, ...
                  'defaultTextFontSize', 8, ...
                  'defaultAxesTickDir',  'out', ...
                  'defaultAxesTickLength', [ 0.02 0 ], ...
                  'defaultAxesFontName', 'helvetica', ...
                  'defaultTextFontName', 'helvetica', ...
                  'defaultAxesLayer', 'top' );

    ax = zeros( nTileX, nTileY );

    for iAx = 1 : nTileX
        for jAx = 1 : nTileY
            ax( iAx, jAx ) = axes( ...
                    'units', 'inches', ...
                    'position', ...
                    [ Fig.deltaX + ( iAx - 1 ) * ( panelX + Fig.gapX ), ...
                      Fig.deltaY + ( nTileY - jAx ) * ( panelY + Fig.gapY ), ...
                      panelX, panelY ] );
        end
    end


    % find initialization time index and construct tick labels for time axis
    tNumPlt = tNumOut; 
    tLabels = cellstr( datestr( tNumPlt, 'mm/yy' ) ); 

    for iX = 1 : nX

        jdxTPltPred = iX + ( 0 : nT - 1 ) * nX;

        for iY = 1 : nTPlt
            
            jF = jdxTPltPred( idxTPltRunning( iY ) );
            jT = ( 1 : nSO ) + idxTPltRunning( iY ) - 1;
            set( gcf, 'currentAxes', ax( iX, iY ) )
            plot( tNumPlt( jT ), fTrue( jF, : ), 'k-' )    
            hold on
            plot( tNumPlt( jT ), fPred( jF, : ), 'b-' )
            plot( tNumPlt( jT ), fPred( jF, : ) + sqrt( fErrPred( jF, : ) ), 'c-' )
            plot( tNumPlt( jT ), fPred( jF, : ) - sqrt( fErrPred( jF, : ) ), 'c-' )
            if iX == 1 && iY == 1
                hL = legend( 'true', 'forecast', 'estimated error', 'location', 'southWest' );
                %hL = legend( 'true', 'forecast', 'location', 'northEast' );
                %lPos = get( hL, 'position' ); 
                %lPos( 2 ) = lPos( 2 ) - .7;
                %set( hL, 'position', lPos )
            end
            %xlabel( '\tau' )
            title( sprintf( 'Lead time \\tau = %i months', idxTPltRunning( iY ) - 1 ) )
            set( gca, 'xLimSpec', 'tight', 'xTick', tNumPlt( 1 : 24 : end ), ...
                'xTickLabel', tLabels( 1 : 12 : end )  )
            if iX == 1 && iY == 1
                ylabel( 'SST anomaly (K)' )
            end
            if iY == nTileY
                xlabel( 'verification time' )
            end
            
        end
    end

    print -dpng -r300 figNino34Running.png
end


if ifPlotProbRunning
    
    nTPlt = numel( idxTPltRunning );

    % Set up figure size
    Fig.figWidth  = 6;    % in inches
    Fig.deltaX    = .5;
    Fig.deltaX2   = .15;
    Fig.deltaY    = .45;
    Fig.deltaY2   = .25;
    Fig.gapX      = .35;
    Fig.gapY      = .5;


    nTileX = nX;
    nTileY = nTPlt;

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 - ( nTileX -1 ) * Fig.gapX ) / nTileX;
    panelY = panelX * ( 3 / 4 ) ^ 3;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Fig.gapX + Fig.deltaX + Fig.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Fig.gapY + Fig.deltaY + Fig.deltaY2 ];


    fig = figure( 'units', 'inches', ...
                  'paperunits', 'inches', ...
                  'position', posn, ...
                  'paperPosition', posn, ...
                  'color', 'white', ...
                  'doubleBuffer', 'on', ...
                  'backingStore', 'off', ...
                  'defaultAxesTickDir', 'out', ...
                  'defaultAxesNextPlot', 'replace', ...
                  'defaultAxesBox', 'on', ...
                  'defaultAxesFontSize', 8, ...
                  'defaultTextFontSize', 8, ...
                  'defaultAxesTickDir',  'out', ...
                  'defaultAxesTickLength', [ 0.02 0 ], ...
                  'defaultAxesFontName', 'helvetica', ...
                  'defaultTextFontName', 'helvetica', ...
                  'defaultAxesLayer', 'top' );

    ax = zeros( nTileX, nTileY );

    for iAx = 1 : nTileX
        for jAx = 1 : nTileY
            ax( iAx, jAx ) = axes( ...
                    'units', 'inches', ...
                    'position', ...
                    [ Fig.deltaX + ( iAx - 1 ) * ( panelX + Fig.gapX ), ...
                      Fig.deltaY + ( nTileY - jAx ) * ( panelY + Fig.gapY ), ...
                      panelX, panelY ] );
        end
    end


    % find initialization time index and construct tick labels for time axis
    tNumPlt = tNumOut; 
    tLabels = cellstr( datestr( tNumPlt, 'mm/yy' ) ); 

    for iX = 1 : nX

        jdxTPltPred = iX + ( 0 : nT - 1 ) * nX;

        for iY = 1 : nTPlt
            
            jF = jdxTPltPred( idxTPltRunning( iY ) );
            jT = ( 1 : nSO ) + idxTPltRunning( iY ) - 1;
            set( gcf, 'currentAxes', ax( iX, iY ) )
            plot( tNumPlt( jT ), chiTrue( jF, : ), 'k-' )    
            hold on
            plot( tNumPlt( jT ), chiPred( jF, : ), 'b-' )
            if iX == 1 && iY == 1
                hL = legend( 'true', 'forecast', 'location', 'southWest' );
                %hL = legend( 'true', 'forecast', 'location', 'northEast' );
                %lPos = get( hL, 'position' ); 
                %lPos( 2 ) = lPos( 2 ) - .7;
                %set( hL, 'position', lPos )
            end
            %xlabel( '\tau' )
            title( sprintf( 'Lead time \\tau = %i months', idxTPltRunning( iY ) - 1 ) )
            set( gca, 'yLim', [ -.1 1.1 ], 'xLimSpec', 'tight', 'xTick', tNumPlt( 1 : 24 : end ), ...
                'xTickLabel', tLabels( 1 : 12 : end )  )
            if iX == 1 && iY == 1
                ylabel( 'probability' )
            end
            if iY == nTileY
                xlabel( 'verification time' )
            end
            
        end
    end

    print -dpng -r300 figNino34ProbRunning.png
end


