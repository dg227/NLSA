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

% Titles for prediction observables
%titleStr = { 'acceleration a_x' 'acceleration a_y' };
titleStr = { 'concentration' };

Plt.idxD = 1;               % observables to plot
Plt.idxT = 1;               % forecast initialization time
Plt.idxX = [ 1 : 10 : nX ]; % x coordinates
Plt.idxY = [ 1 : 2 : nY ];  % y coordinates


%% SCRIPT EXCECUTION OPTIONS
ifRead        = true; % read data and kernel eigenfunctions 
ifPred        = true; % perform prediction 
ifErr         = true; % compute prediction error 
ifVar         = true; % predict variance

ifPlotPred = true; % plot forecasts at representative grid points
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
    fTau = permute( fTau, [ 1 3 4 2 ] ); % size( fTau ) = [ nD nP nT nG ]
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

    % put test data in appropriate form for time shift (temporal index is
    % last) 
    fT = permute( fOut, [ 1 3 2 ] );
    fT = reshape( fT, [ nD * nG, nTO + nTA ] );

    % create true signal by time-shifting out-of-sample data
    fTrue = lembed( fT, [ nP, nP + nTO - 1 ], 1 : nP );

    % put in appropriate form for comparison with fPred 
    fTrue = reshape( fTrue, [ nD nG nP nTO ] );
    fTrue = permute( fTrue, [ 1 3 4 2 ] ); % size( fTrue ) = [ nD nP nTO nG ]
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

if ifPlotErr

    % Set up figure size
    Fig.figWidth  = 6;    % in inches
    Fig.deltaX    = .57;
    Fig.deltaX2   = .1;
    Fig.deltaY    = .48;
    Fig.deltaY2   = .24;
    Fig.gapX      = .35;
    Fig.gapY      = .3;


    nTileX = nD;
    nTileY = 1;

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 - ( nTileX -1 ) * Fig.gapX ) / nTileX;
    panelY = panelX * 9 / 16;

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

    tVals = dt * ( 0 : nP - 1 );

    for iD = 1 : nD

        set( gcf, 'currentAxes', ax( iD, 1 ) )
        plot( tVals, fRmse( iD, : ), 'k-' )
        hold on 
        plot( tVals, fRmsePred( iD, : ), 'b-' ) 
        set( gca, 'xLimSpec', 'tight' )
        set( gca, 'yLim', [ 0 1.2 ], 'yTick', [ 0 : .2 : 1.2 ] )
        grid on
        xlabel( 'lead time t' )
        if iD == 1
            ylabel( 'normalized L2 error' )
        end

        title( titleStr{ iD } )

        if iD == 1
            legend( 'true error', 'estimated error', 'location', 'southEast' )
        end
    end

    print( '-dpng', '-r300', [ 'fig_' experiment '_err.png' ] )
end


if ifPlotPred

    % Prepare variabes to plot
    tVals = dt * ( 0 : nP - 1 );

    fTruePlt    = reshape( fTrue, [ nD nP nTO nY nX ] );
    fPredPlt    = reshape( fPred, [ nD nP nTO nY nX ] ); 
    fErrPredPlt = reshape( fErrPred, [ nD nP nTO nY nX ] );

    fTruePlt = squeeze( ...
        fTruePlt( Plt.idxD, :, Plt.idxT, Plt.idxY, Plt.idxX ) );
    fPredPlt = squeeze( ...
        fPredPlt( Plt.idxD, :, Plt.idxT, Plt.idxY, Plt.idxX ) );
    fErrPredPlt = squeeze( ...
        fErrPredPlt( Plt.idxD :, Plt.idxT, Plt.idxY, Plt.idxX ) );
 
    % Set up figure and axes size parameters
    Fig.figWidth = 6;    % in inches
    Fig.deltaX   = .57;
    Fig.deltaX2  = .1;
    Fig.deltaY   = .48;
    Fig.deltaY2  = .12;
    Fig.gapX     = .35;
    Fig.gapY     = .3;
    Fig.nTileX   = numel( Plt.idxX );
    Fig.nTileY   = numel( Plt.idxY );
    Fig.aspectR  = 9 / 16;

    % Loop over the response variables
    nDPlt = numel( Plt.idxD );
    fig = zeros( 1, nDPlt );
    for iD = 1 : nDPlt
        iDPlt = Plt.idxD( iD );
        fig( iD ) = plotPred( squeeze( fTruePlt( iDPlt, :, :, :, : ) ), ...
                              squeeze( fPredPlt( iDPlt, :, :, : : ) ), ...
                              squeeze( fPredErrPlt( iDPlt, :, :, :, : ) ), ...
                              Plt, Fig, x, y, dt, varStr{ iD } );


        print( '-dpng', '-r300', ...
            [ 'fig_' experiment '_' varStr{ iD } '.png' ] )
    end 
end


% Function to plot predictions

function fig = plotPred( fTrue, fPred, fPredErr, Plt, Fig, x, y, varStr );

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 ...
               - ( Fig.nTileX -1 ) * Fig.gapX ) / Fig.nTileX;
    panelY = panelX * Fig.aspectR;

    posn = [ 0 ...
             0 ...
             Fig.nTileX * panelX + ( Fig.nTileX - 1 ) * Fig.gapX ...
                 + Fig.deltaX + Fig.deltaX2 ...
             Fig.nTileY * panelY + ( Fig.nTileY - 1 ) * Fig.gapY ...
                 + Fig.deltaY + Fig.deltaY2 ];


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

    ax = zeros( Fig.nTileX, Fig.nTileY );

    for iAx = 1 : Fig.nTileX
        for jAx = 1 : Fig.nTileY
            ax( iAx, jAx ) = axes( ...
                'units', 'inches', ...
                'position', ...
                [ Fig.deltaX + ( iAx - 1 ) * ( panelX + Fig.gapX ), ...
                  Fig.deltaY + ( Fig.nTileY - jAx ) * ( panelY + Fig.gapY ), ...
                  panelX, panelY ] );
        end
    end

    axTitle = axes( ...
        'units', 'inches', ...
        'position', ...
        [ Fig.deltaX, Fig.deltaY, ...
          Fig.nTileX * Fig.panelX + ( Fig.nTileX - 1 ) * Fig.gapX, ...
          Fig.nTileY * Fig.panelY + ( Fig.nTileY - 1 ) * Fig.gapY + 15 ], ...
        'color', 'none', ...
        'box', 'off' );
   
    tVals = ( 0 : nP - 1 ) * dt;

    for iY = 1 : Fig.nTileY
        for iX = 1 : Fig.nTileX 

            set( gcf, 'currentAxes', ax( iX, iY ) )
            [ hPred, hErr ] = boundedline( tVals, ...
                fPred( :, Plt.idxY( iY ), Plt.idxX( iX ) ), ...
                sqrt( fErrPred( :, Plt.idxY( iY ), Plt.idxX( iX ) ) ), 'r-' );
            hold on
            hTrue = plot( tVals, ...
                          fTrue( :, Plt.idxY( iY ), Plt.idxX( iX ) ), 'k-' );    
            if iY == 1 && iX == 1
                hL = legend( [ hTrue hPred ],  'true', 'forecast', ... 
                             'location', 'southEast' );
                %lPos = get( hL, 'position' ); 
                %lPos( 1 ) = lPos( 1 ) - .02;
                %lPos( 2 ) = lPos( 2 ) - .75;
                %set( hL, 'position', lPos )
            end

            if iX ~= 1
                set( gca, 'yTickLabel', [] )
            end

            if iY == 1
                xlabel( 'lead time (hours)' )
            else
                set( gca, 'xTickLabel', [] )
            end

            set( gca, 'xLimSpec', 'tight' )

            title( sprintf( '(x,y) = (%1.2f,%1.2f) km' ) )

        end
    end

    title( axTitle, sprintf( '%s, initialization time %1.2f hours', ...
        varStr, dt * ( Plt.idxTO - 1 ) ) )

        % bring legend to foreground
%        aax = get( gcf, 'children' );
%        ind = find( isgraphics( aax, 'Legend' ) );
%        set( gcf, 'children', aax( [ ind : end, 1 : ind - 1 ] ) )
        %legend( 'true error', 'estimated error', 'location', 'northwest' )
        %legend boxoff

end







