% DEMO OF NLSA APPLIED TO LORENZ 63 DATA
%
% Modified 2020/06/06

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
%experiment = '6.4k_dt0.01_nEL0'; % 6400 samples, sampling interval 0.01, 0 delays 
%experiment = '6.4k_dt0.01_nEL80'; % 6400 samples, sampling interval 0.01, 80 delays 
%experiment = '6.4k_dt0.01_nEL100'; % 6400 samples, sampling interval 0.01, 100 delays 
%experiment = '6.4k_dt0.01_nEL150'; % 6400 samples, sampling interval 0.01, 150 delays 
%experiment = '6.4k_dt0.01_nEL200'; % 6400 samples, sampling interval 0.01, 200 delays 
%experiment = '6.4k_dt0.01_nEL300'; % 6400 samples, sampling interval 0.01, 300 delays 
%experiment = '6.4k_dt0.01_nEL400'; % 6400 samples, sampling interval 0.01, 400 delays 
%experiment = '64k_dt0.01_nEL0'; % 64000 samples, sampling interval 0.01, no delays 
%experiment = '64k_dt0.01_nEL400'; % 64000 samples, sampling interval 0.01, 400 delays
experiment = '64k_dt0.01_nEL800'; % 64000 samples, sampling interval 0.01, 800 delays
%experiment = '64k_dt0.01_nEL1600'; % 64000 samples, sampling interval 0.01, 1600 delays
%experiment = '64k_dt0.01_nEL3200'; % 64000 samples, sampling interval 0.01, 3200 delays

ifSourceData   = false; % generate source data
ifNLSA         = false; % run NLSA
ifPlotPhi      = false; % plot eigenfunctions
ifMoviePhi     = true;  % make eigenfunction movie
ifPrintFig     = true;  % print figures to file

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% GLOBAL PARAMETERS
% nShiftPlt:   Temporal shift applied to eigenfunctions to illustrate action
%              of Koopman operator
% idxPhiPlt:   Eigenfunctions to plot
% idxTPlt:     Time interval to plot
% figDir:      Output directory for plots
% markerSize:  For eigenfunction scatterplots

switch experiment

% 6400 samples, sampling interval 0.01, no delay embedding 
case '6.4k_dt0.01_nEL0'

    idxPhiPlt  = [ 2 3 5 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         

% 6400 samples, sampling interval 0.01, no delay embedding 
case '6.4k_dt0.01_nEL80'

    idxPhiPlt  = [ 4 5 8 9 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         


% 6400 samples, sampling interval 0.01, 100 delays
case '6.4k_dt0.01_nEL100'

    idxPhiPlt  = [ 2 3 4 5 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         

% 6400 samples, sampling interval 0.01, 100 delays
case '6.4k_dt0.01_nEL150'

    idxPhiPlt  = [ 2 3 4 5 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         

% 6400 samples, sampling interval 0.01, 200 delays
case '6.4k_dt0.01_nEL200'

    idxPhiPlt  = [ 2 3 4 5 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         

% 6400 samples, sampling interval 0.01, 300 delays
case '6.4k_dt0.01_nEL300'

    idxPhiPlt  = [ 2 3 4 5  ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         

% 6400 samples, sampling interval 0.01, 400 delays
case '6.4k_dt0.01_nEL400'

    idxPhiPlt  = [ 2 3 4 5  ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         

% 64000 samples, sampling interval 0.01, no delay embedding 
case '64k_dt0.01_nEL0'

    idxPhiPlt  = [ 2 3 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 3;         

% 64000 samples, sampling interval 0.01, 400 delays
case '64k_dt0.01_nEL400'

    idxPhiPlt  = [ 2 3 4 5  ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 3;         

% 64000 samples, sampling interval 0.01, 800 delays
case '64k_dt0.01_nEL800'

    idxPhiPlt  = [ 2 3 ];     
    signPhiPlt = [ 1 -1 ];   
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3001 ]; % approx 10 Lyapunov timescales
    markerSize = 3;         

% 64000 samples, sampling interval 0.01, 1600 delays
case '64k_dt0.01_nEL1600'

    idxPhiPlt  = [ 2 3  ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3001 ]; % approx 10 Lyapunov timescales
    markerSize = 3;         

% 64000 samples, sampling interval 0.01, 3200 delays
case '64k_dt0.01_nEL3200'

    idxPhiPlt  = [ 2 3 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 3;         


end

% Figure directory
figDir = fullfile( pwd, 'figs', experiment );
if ~isdir( figDir )
    mkdir( figDir )
end

%% EXTRACT SOURCE DATA
if ifSourceData
    disp( 'Generating source data...' ); t = tic;
    demoNLSA_data( experiment ) 
    toc( t )
end

%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES
% In is a data structure containing the NLSA parameters for the training data.
%
% nSE is the number of samples avaiable for data analysis after Takens delay
% embedding.
%
% nSB is the number of samples left out in the start of the time interval (for
% temporal finite differnences employed in the kerenl).
%
% nEL is the Takens embedding window length (in number of timesteps)
%
% nShiftTakens is the temporal shift applied to align eigenfunction data with 
% the center of the Takens embedding window. 

disp( 'Building NLSA model...' ); t = tic;
[ model, In ] = demoNLSA_nlsaModel( experiment ); 
toc( t )

nSE          = getNTotalSample( model.embComponent );
nSB          = getNXB( model.embComponent );
nEL          = getEmbeddingWindow( model.embComponent ) - 1;
nShiftTakens = round( nEL / 2 );

%% PERFORM NLSA
if ifNLSA
    
    % Execute NLSA steps. Output from each step is saved on disk.

    disp( 'Takens delay embedding...' ); t = tic; 
    computeDelayEmbedding( model )
    toc( t )


    fprintf( 'Pairwise distances for density data, %i/%i...\n', iProc, nProc ); 
    t = tic;
    computeDenPairwiseDistances( model, iProc, nProc )
    toc( t )

    disp( 'Distance normalization for KDE...' ); t = tic;
    computeDenBandwidthNormalization( model );
    toc( t )

    disp( 'Kernel tuning for KDE...' ); t = tic;
    computeDenKernelDoubleSum( model );
    toc( t )

    disp( 'Kernel density estimation...' ); t = tic;
    computeDensity( model );
    toc( t )

    disp( 'Takens delay embedding for density data...' ); t = tic;
    computeDensityDelayEmbedding( model );
    toc( t )

    fprintf( 'Pairwise distances (%i/%i)...\n', iProc, nProc ); t = tic;
    computePairwiseDistances( model, iProc, nProc )
    toc( t )

    disp( 'Distance symmetrization...' ); t = tic;
    symmetrizeDistances( model )
    toc( t )

    disp( 'Kernel tuning...' ); t = tic;
    computeKernelDoubleSum( model )
    toc( t )

    disp( 'Kernel eigenfunctions...' ); t = tic;
    computeDiffusionEigenfunctions( model )
    toc( t )

end

%% PLOT EIGENFUNCTIONS
if ifPlotPhi
    
    % Retrieve source data and NLSA eigenfunctions. Assign timestamps.
    x = getData( model.srcComponent );
    x = x( :, 1 + nShiftTakens : nSE + nShiftTakens );
    [ phi, ~, lambda ] = getDiffusionEigenfunctions( model );
    t = ( 0 : nSE - 1 ) * In.dt;  


    % Set up figure and axes 
    Fig.nTileX     = numel( nShiftPlt ) + 1;
    Fig.nTileY     = numel( idxPhiPlt );
    Fig.units      = 'inches';
    Fig.figWidth   = 15 / 4 * Fig.nTileX; 
    Fig.deltaX     = .2;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = 0.4;
    Fig.gapT       = 0.4; 
    Fig.aspectR    = 3 / 4;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % EIGENFUNCTION SCATTERPLOTS

    % Loop over the time shifts
    for iShift = 1 : Fig.nTileX - 1

        xPlt = x( :, 1 : end - nShiftPlt( iShift ) );

        % Loop over the eigenfunctions
        for iPhi = 1 : Fig.nTileY

            phiPlt = phi( 1 + nShiftPlt( iShift ) : end, idxPhiPlt( iPhi ) );

            set( gcf, 'currentAxes', ax( iShift, iPhi ) )
            scatter3( xPlt( 1, : ), xPlt( 2, : ), xPlt( 3, : ), markerSize, ...
                      phiPlt, 'filled'  )
            axis off
            view( 0, 0 )
            set( gca, 'cLim', max( abs( phiPlt ) ) * [ -1 1 ] )
            
            if iShift == 1
                titleStr = sprintf( '\\phi_{%i}, \\lambda_{%i} = %1.3g', ...
                                    idxPhiPlt( iPhi ), idxPhiPlt( iPhi ), ...
                                    lambda( idxPhiPlt( iPhi ) ) );
            else
                titleStr = sprintf( 'U^t\\phi_{%i},   t = %1.2f', ...
                                    idxPhiPlt( iPhi ), ...
                                    nShiftPlt( iShift ) * In.dt ); 
            end
            title( titleStr )
        end

    end

    % EIGENFUNCTION TIME SERIES PLOTS

    tPlt = t( idxTPlt( 1 ) : idxTPlt( 2 ) );
    tPlt = tPlt - tPlt( 1 ); % set time origin to 1st plotted point

    % Loop over the eigenfunctions
    for iPhi = 1 : Fig.nTileY

        phiPlt = phi( idxTPlt( 1 ) : idxTPlt( 2 ), idxPhiPlt( iPhi ) );

        set( gcf, 'currentAxes', ax( Fig.nTileX, iPhi ) )
        plot( tPlt, phiPlt, '-' )
        grid on
        xlim( [ tPlt( 1 ) tPlt( end ) ] )
        ylim( [ -3 3 ] )

        if iPhi == 1
            title( 'Time series along orbit' )
        end
        if iPhi == Fig.nTileY
            xlabel( 't' )
        end
    end

    titleStr = [ sprintf( 'Sampling interval \\Deltat = %1.2f, ', In.dt ) ...
                 sprintf( 'Delay embedding window T = %1.2f', In.dt * nEL ) ]; 
    title( axTitle, titleStr )


    % Print figure
    if ifPrintFig
        figFile = sprintf( 'figPhi%s.png', idx2str( idxPhiPlt, '_' ) );
        figFile = fullfile( figDir, figFile );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end



%% MAKE EIGENFUNCTION MOVIE
if ifMoviePhi
    
    % Retrieve source data and NLSA eigenfunctions
    x = getData( model.srcComponent );
    x = x( :, 1 + nShiftTakens : nSE + nShiftTakens );
    [ phi, ~, lambda ] = getDiffusionEigenfunctions( model );


    % Set up figure and axes 
    Fig.nTileX     = 3;
    Fig.nTileY     = 2;
    Fig.units      = 'pixels';
    Fig.figWidth   = 1000; 
    Fig.deltaX     = 20;
    Fig.deltaX2    = 20;
    Fig.deltaY     = 50;
    Fig.deltaY2    = 30;
    Fig.gapX       = 50;
    Fig.gapY       = 70;
    Fig.gapT       = 40; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Set up videowriter
    movieFile = sprintf( 'moviePhi%s.png', idx2str( idxPhiPlt, '_' ) );
    movieFile = fullfile( figDir, movieFile );
    writerObj = VideoWriter( movieFile, 'MPEG-4' );
    writerObj.FrameRate = 20;
    writerObj.Quality = 100;
    open( writerObj );

    % Determine number of movie frames; assign timestamps
    nFrame = idxTPlt( 2 ) - idxTPlt( 1 ) + 1;
    t = ( 0 : nFrame - 1 ) * In.dt;  

    % Construct coherent observable
    z = phi( :, idxPhiPlt ) .* signPhiPlt / sqrt( 2 );

    % Loop over the frames
    for iFrame = 1 : nFrame

        iT = idxTPlt( 1 ) + iFrame - 1;

        % Scatterplot of x2 observable
        set( gcf, 'currentAxes', ax( 1, 1 ) )
        scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, x( 2, : ), ...
                  'filled'  )
        scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
        title( 'x_2 observable' )
        axis off
        view( 0, 0 )
        set( gca, 'cLim', max( abs( x( 2, : ) ) ) * [ -1 1 ] ) 

        % (x1,x2) projection
        set( gcf, 'currentAxes', ax( 2, 1 ) )
        plot( x( 1, idxTPlt( 1 ) : iT ), x( 2, idxTPlt( 1 ) : iT ), 'b-', ...
              'linewidth', 1.5 ) 
        scatter( x( 1, iT ), x( 2, iT ), 50, 'r', 'filled' ) 
        xlim( [ -30 30 ] )
        ylim( [ -30 30 ] )
        grid on
        title( '(x_1,x_2) projection' )
        xlabel( 'x_1' )
        ylabel( 'x_2' )

        
        % x2 time series
        set( gcf, 'currentAxes', ax( 3, 1 ) )
        plot( t( 1 : iFrame ), x( 2, idxTPlt( 1 ) : iT ), 'b-', ...
              'linewidth', 1.5 )
        scatter( t( iFrame ), x( 2, iT ), 50, 'r', 'filled' ) 
        grid on
        xlim( [ t( 1 ) t( end ) ] )
        ylim( [ -30 30 ] )
        title( 'x_2 time series' )
        xlabel( 't' )
        ylabel( 'x_2' )

        % phi2 scatterplot
        set( gcf, 'currentAxes', ax( 1, 2 ) )
        scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, z( :, 2 ), ...
                  'filled'  )
        scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
        title( sprintf( '\\phi_{%i} observable', idxPhiPlt( 2 ) - 1 ) )
        axis off
        view( 0, 0 )
        set( gca, 'cLim', max( abs( z( :, 2 ) ) ) * [ -1 1 ] )

        % (phi1,phi2) projection
        set( gcf, 'currentAxes', ax( 2, 2 ) )
        plot( z( idxTPlt( 1 ) : iT, 1 ), z( idxTPlt( 1 ) : iT, 2 ), ...
              'b-', 'linewidth', 1.5 ) 
        scatter( z( iT, 1 ), z( iT, 2 ), 50, 'r', 'filled' ) 
        xlim( [ -1.5 1.5 ] )
        ylim( [ -1.5 1.5 ] )
        grid on
        title( sprintf( '(\\phi_{%i},\\phi_{%i}) projection', ...
               idxPhiPlt( 1 ) - 1, idxPhiPlt( 2 ) - 1 ) )
        xlabel( sprintf( '\\phi_{%i}', idxPhiPlt( 1 ) - 1 ) )
        ylabel( sprintf( '\\phi_{%i}', idxPhiPlt( 2 ) - 1 ) )

        % phi2 time series
        set( gcf, 'currentAxes', ax( 3, 2 ) )
        plot( t( 1 : iFrame ), z( idxTPlt( 1 ) : iT, 2 ), 'b-', ...
              'linewidth', 1.5 )
        scatter( t( iFrame ), z( iT, 2 ), 50, 'r', 'filled' ) 
        grid on
        xlim( [ t( 1 ) t( end ) ] )
        ylim( [ -1.5 1.5 ] )
        title( sprintf( '\\phi_{%i} time series', idxPhiPlt( 2 ) - 1 ) )
        ylabel( sprintf( '\\phi_{%i}', idxPhiPlt( 2 ) - 1 ) )
        xlabel( 't' )

        title( axTitle, sprintf( 't = %1.2f', t( iFrame ) ) )
        axis( axTitle, 'off' )
        
        frame = getframe( fig );
        writeVideo( writerObj, frame )
        
        for iY = 1 : Fig.nTileY
            for iX = 1 : Fig.nTileX
                cla( ax( iX, iY ), 'reset' )
            end
        end
        cla( axTitle, 'reset' )
    end

    % Close video file and figure
    close( writerObj )
    close( fig )
end

        
    
