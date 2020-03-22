% KERNEL ANALOG FORECASTING OF SEA ICE DYNAMICS
%
% Modified 2020/03/19

%% MAIN CALCULATION PARAMETERS AND OPTIONS
experiment = 'channel_cuvocn'; 
idxPhi     = 1 : 101;  % kernel eigenfunctions 
nP         = 15 + 1;   % prediction timesteps (including 0)

xLim =  [ -2.5E5 2.5E5 ]; 
yLim =  [ -5E4 5E4 ]; 
nX = 25;  % number of coarse cells, x direction
nY = 5;   % number of coarse cells, y direction

Dem.dt  = 20; % DEM time step (sec)
Dem.nDT = 50; % number of DEM timesteps between output

dt = Dem.dt * Dem.nDT / 3600; % Forecast time step (hours)


%% SCRIPT EXCECUTION OPTIONS
ifRead        = true; % read data and kernel eigenfunctions 
ifPred        = true; % perform prediction 
ifErr         = true; % compute prediction error 
ifVar         = true; % predict variance

ifPlotPred = false; % plot forecasts at representative grid points
ifPlotErr  = false; % plot error metrics

%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES
[ model, Pars ] = floeVSAModel_den_ose( experiment );

nPhi = numel( idxPhi );                                  % basis functions
nS   = getNTotalSample( model.embComponent( 1, : ) );    % training samples
nSO  = getNTotalSample( model.outEmbComponent( 1, : ) ); % test samples  
nSA  = sum( getNXA( model.trgEmbComponent( 1, : ) ) );   % extra samples
nD   = sum( getDataSpaceDimension( model.trgEmbComponent( :, 1 ) ) ); 

nG  = nX * nY;  % spatial gridpoints
nT  = nS / nG;  % temporal training samples  
nTO = nSO / nG; % temporal test samples
nTA = nSA / nG; % extra temporal samples (to construct time-shifted response)

%% READ DATA FROM MODEL
if ifRead
    tic
    disp( 'Data retrieval')

    % in-sample eigenfunctions 
    [ phi, mu, lambda ] = getDiffusionEigenfunctions( model ); 

    % out-of-sample eigenfunctions
    phiO = getOseDiffusionEigenfunctions( model );


    % training data 
    f = zeros( nD, nT + nTA, nG ); 
    f( :, 1 : nT, : ) = reshape( ...
        getData( model.trgEmbComponent ), [ nD nT nG ] );
    f( :, nT + 1 : end, : ) = reshape( ...
        getData_after( model.trgEmbComponent ), [ nD nTA nG ] );

    % compute space-time mean and standard deviation
    fMean = mean( f, [ 2  3 ] );   
    fStd  = std( f, 0, [ 2  3 ] );

    % verification data 
    fOut = zeros( nD, nTO + nTA, nG ); 
    fOut( :, 1 : nTO, : ) = reshape( ...
        getData( model.outTrgEmbComponent ), [ nD nTO nG ] );
    fOut( :, nTO + 1 : end, : ) = reshape( ...
        getData_after( model.outTrgEmbComponent ), [ nD nTA nG ] );

    toc
end


if ifPred
    tic
    disp( 'Prediction' )

    % put training data in appropriate form for time shift (temporal index is
    % last) 
    fT = permute( f, [ 1 3 2 ] );
    fT = reshape( fT, [ nD * nG, nT + nTA ] );

    % create time-shifted copies of the input signal
    fTau = lembed( fT, [ nP, nP + nT - 1 ], 1 : nP );

    % put in appropriate form for multiplication with basis functions (phi)
    fTau = reshape( fTau, [ nD nG nP nT ] );
    fTau = permute( fTau, [ 1 3 4 2 ] );
    fTau = reshape( fTau, [ nD * nP, nT * nG ] );

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

    % put training data in appropriate form for time shift (temporal index is
    % last) 
    fT = permute( fOut, [ 1 3 2 ] );
    fT = reshape( fT, [ nD * nG, nTO + nTA ] );

    % create true signal by time-shifting out-of-sample data
    fTrue = lembed( fT, [ nP, nP + nTO - 1 ], 1 : nP );

    % put in appropriate form for comparison with fPred 
    fTrue = reshape( fTrue, [ nD nG nP nTO ] );
    fTrue = permute( fTrue, [ 1 3 4 2 ] );
    fTrue = reshape( fTrue, [ nD * nP, nTO * nG ] );

    % compute normalized RMSE
    predErr = fPred - fTrue;
    fRmse = vecnorm( predErr, 2, 2 ) / sqrt( nSO );
    fRmse = bsxfun( @rdivide, reshape( fRmse, [ nD nP ] ), fStd ); 

    toc
end

if ifVar
    tic
    disp( 'Forecast variance' )
    
    % in-sample error
    %fErr = abs( fTau - cTau * phi( :, idxPhi )' );
    fErr = ( fTau - cTau * phi( :, idxPhi )' ) .^ 2;

    % compute regression coefficients of in-sample error against 
    % eigenfunctions
    c2Tau = fErr * bsxfun( @times, phi( :, idxPhi ), mu ); 
     
    % evaluate error forecast
    fErrPred = abs( c2Tau * phiO( :, idxPhi )' );
    toc

    % compute predicted RMSE
    %fRmsePred = vecnorm( fErrPred, 2, 2 ) / sqrt( nSO );
    fRmsePred = sqrt( vecnorm( fErrPred, 1, 2 ) / nSO );
    fRmsePred = bsxfun( @rdivide, reshape( fRmsePred, [ nD nP ] ), fStd );
end

    
if ifPlotPred

    % Set up figure size
    Fig.figWidth  = 6;    % in inches
    Fig.deltaX    = .57;
    Fig.deltaX2   = .1;
    Fig.deltaY    = .48;
    Fig.deltaY2   = .12;
    Fig.gapX      = .35;
    Fig.gapY      = .3;


    nTileX = 1;
    nTileY = 2;

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 - ( nTileX -1 ) * Fig.gapX ) / nTileX;
    panelY = panelX * 9 / 16 * .9;

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
                  'defaultAxesFontSize', 12, ...
                  'defaultTextFontSize', 12, ...
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

    for iX = 1 : 1

        tVals = ( idxTPltPred - 1 ) * Pars.dt;
        jdxTPltPred = iX + ( 0 : nP - 1 ) * 3;

        set( gcf, 'currentAxes', ax( iX, 1 ) )
        [ hPredPlt, hErrPlt ] = boundedline( tVals, fPred( jdxTPltPred, idxT0Plt ), sqrt( fErrPred( jdxTPltPred, idxT0Plt ) ), 'r-' );
        hold on
        [ hPredRefPlt, hErrRefPlt ] = boundedline( tVals, fPredRef( jdxTPltPred, idxT0Plt ), sqrt( fErrPredRef( jdxTPltPred, idxT0Plt ) ), 'b-' );
        hTruePlt = plot( tVals, fTrue( jdxTPltPred, idxT0Plt ), 'k-' );    
        plot( tVals, fPred( jdxTPltPred, idxT0Plt ), 'r-' )
        %plot( tVals, fPredRef( jdxTPltPred, idxT0Plt ) ...
        %            + , 'b--' )
        %plot( tVals, fPredRef( jdxTPltPred, idxT0Plt ) ...
        %            - sqrt( fErrPredRef( jdxTPltPred, idxT0Plt ) ), 'b--' )
        %plot( tVals, fPred( jdxTPltPred, idxT0Plt ) ...
        %            + sqrt( fErrPred( jdxTPltPred, idxT0Plt ) ), 'r--' )
        %plot( tVals, fPred( jdxTPltPred, idxT0Plt ) ...
        %            - sqrt( fErrPred( jdxTPltPred, idxT0Plt ) ), 'r--' )
        if iX == 1
            hL = legend( [ hTruePlt hPredRefPlt hPredPlt ],  'true', 'full obs.', '\omega^1 only', 'location', 'northEast' );
            lPos = get( hL, 'position' ); 
            lPos( 1 ) = lPos( 1 ) - .02;
            lPos( 2 ) = lPos( 2 ) - .75;
            set( hL, 'position', lPos )
        end
        %xlabel( 't' )
        ylabel( sprintf( '\\omega^%i', iX ) )
        set( gca, 'xLimSpec', 'tight' )

        set( gcf, 'currentAxes', ax( iX, 2 ) )
        plot( tVals, fRmse( iX, idxTPltPred ), 'r-' )
        hold on 
        plot( tVals, fRmsePred( iX, idxTPltPred ), 'r--' ) 
        plot( tVals, fRmseRef( iX, idxTPltPred ), 'b-' )
        plot( tVals, fRmsePredRef( iX, idxTPltPred ), 'b--' ) 
        set( gca, 'xLimSpec', 'tight' )
        set( gca, 'yLim', [ 0 1.2 ], 'yTick', [ 0 : .2 : 1.2 ] )
        grid on
        xlabel( 'lead time t' )
        if iX == 1
            ylabel( 'normalized L2 error' )
        end

        % bring legend to foreground
        aax = get( gcf, 'children' );
        ind = find( isgraphics( aax, 'Legend' ) );
        set( gcf, 'children', aax( [ ind : end, 1 : ind - 1 ] ) )
        %legend( 'true error', 'estimated error', 'location', 'northwest' )
        %legend boxoff
    end

    print -dpng -r300 figL63Pred.png
end







