% DEMO OF NLSA APPLIED TO LORENZ 63 DATA
%
% Modified 2020/06/06

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
%experiment = '6.4k_dt0.01_nEL0'; % 6400 samples, sampling interval 0.01, 0 delays 
%experiment = '6.4k_dt0.01_nEL10'; % 6400 samples, sampling interval 0.01, 10 delays 
%experiment = '6.4k_dt0.01_nEL80'; % 6400 samples, sampling interval 0.01, 80 delays 
%experiment = '6.4k_dt0.01_nEL100'; % 6400 samples, sampling interval 0.01, 100 delays 
%experiment = '6.4k_dt0.01_nEL150'; % 6400 samples, sampling interval 0.01, 150 delays 
%experiment = '6.4k_dt0.01_nEL200'; % 6400 samples, sampling interval 0.01, 200 delays 
%experiment = '6.4k_dt0.01_nEL300'; % 6400 samples, sampling interval 0.01, 300 delays 
%experiment = '6.4k_dt0.01_nEL400'; % 6400 samples, sampling interval 0.01, 400 delays 
%experiment = '6.4k_dt0.01_nEL800'; % 6400 samples, sampling interval 0.01, 400 delays 
%experiment = '64k_dt0.01_nEL0'; % 64000 samples, sampling interval 0.01, no delays 
%experiment = '64k_dt0.01_nEL400'; % 64000 samples, sampling interval 0.01, 400 delays
experiment = '64k_dt0.01_nEL800'; % 64000 samples, sampling interval 0.01, 800 delays
%experiment = '64k_dt0.01_nEL1600'; % 64000 samples, sampling interval 0.01, 1600 delays
%experiment = '64k_dt0.01_nEL3200'; % 64000 samples, sampling interval 0.01, 3200 delays

ifSourceData     = false; % generate source data
ifNLSA           = false; % run NLSA
ifPCA            = true;  % run PCA (for comparison with NLSA)
ifPlotPhi        = false; % plot eigenfunctions
ifPlotCoherence  = true; % figure illustrating coherence of NLSA eigenfunctions
ifMovieCoherence = false;  % make eigenfunction movie
ifPrintFig       = true;  % print figures to file
ifPool           = false;  % create parallel pool

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
    idxTPlt    = [ 2001 3001 ]; % approx 10 Lyapunov timescales
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
    signPC = [ -1 1 ];

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

% Create parallel pool if required
% In.nParE is the number of parallel workers for delay-embedded distances
% In.nParNN is the number of parallel workers for nearest neighbor search
if ifPool
    if isfield( In, 'nParE' ) && In.nParE > 0
        nPar = In.nParE;
    else
        nPar = 0;
    end
    if isfield( In, 'nParNN' ) && In.nParNN > 0
        nPar = max( nPar, In.nParNN );
    end
    if nPar > 0
        poolObj = gcp( 'nocreate' );
        if isempty( poolObj )
            poolObj = parpool( nPar );
        end
    end
end

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

    disp( 'Distance normalization for kernel density steimation...' ); t = tic;
    computeDenBandwidthNormalization( model );
    toc( t )

    disp( 'Kernel bandwidth tuning for density estimation...' ); t = tic;
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

    disp( 'Kernel bandwidth tuning...' ); t = tic;
    computeKernelDoubleSum( model )
    toc( t )

    disp( 'Kernel eigenfunctions...' ); t = tic;
    computeDiffusionEigenfunctions( model )
    toc( t )

end

%% PERFORM PCA
if ifPCA

    disp( 'PCA...' ); t = tic;
    x = getData( model.srcComponent );
    [ PCA.u, PCA.s, PCA.v ] = svd( x - mean( x, 2 ), 'econ' );
    PCA.v = PCA.v * sqrt( getNTotalSample( model.srcComponent ) );
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
                                    idxPhiPlt( iPhi ) - 1, ...
                                    idxPhiPlt( iPhi ) - 1, ...
                                    lambda( idxPhiPlt( iPhi ) ) );
            else
                titleStr = sprintf( 'U^t\\phi_{%i},   t = %1.2f', ...
                                    idxPhiPlt( iPhi ) - 1, ...
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


%% COHERENCE FIGURE
if ifPlotCoherence
    
    % Retrieve source data and NLSA eigenfunctions
    x = getData( model.srcComponent );
    x = x( :, 1 + nShiftTakens : nSE + nShiftTakens );
    [ phi, ~, lambda ] = getDiffusionEigenfunctions( model );

    % Construct coherent observables based on phi's
    z = phi( :, idxPhiPlt ) .* signPhiPlt / sqrt( 2 );
    a = max( abs( z ), [], 1 ); % z amplitude
    angl = angle( complex( z( :, 1 ), z( :, 2 ) ) ); % z phase

    % Construct coherent observables based on PCs
    zPC = PCA.v( 1 + nShiftTakens : nSE + nShiftTakens, [ 1 2 ] ) ...
        .* signPC / sqrt( 2 );
    anglPC = angle( complex( zPC( :, 1 ), zPC( :, 2 ) ) ); % z phase
    aPC = max( abs( zPC ), [], 1 ); % PC amplitude 

    % Determine number of temporal samples; assign timestamps
    nFrame = idxTPlt( 2 ) - idxTPlt( 1 ) + 1;
    t = ( 0 : nFrame - 1 ) * In.dt;  

    % Set up figure and axes 
    Fig.nTileX     = 4;
    Fig.nTileY     = 2;
    Fig.units      = 'inches';
    Fig.figWidth   = 15 / 4 * Fig.nTileX; 
    Fig.deltaX     = .4;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .6;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = 1;
    Fig.gapT       = 0; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );
    set( fig, 'invertHardCopy', 'off' )

    % Scatterplot of PC1
    set( gcf, 'currentAxes', ax( 1, 1 ) )
    scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, zPC( :, 1 ), ...
              'filled'  )
    plot3( x( 1, idxTPlt( 1 ) : idxTPlt( 2 ) ), ...
           x( 2, idxTPlt( 1 ) : idxTPlt( 2 ) ), ...
           x( 3, idxTPlt( 1 ) : idxTPlt( 2 ) ), 'k-', ...
           'lineWidth', 2 )
              
    title( '(a) PC_1 on L63 attracror' )
    axis off
    view( 0, 0 )
    set( gca, 'cLim', [ -1 1 ] * aPC( 1 ) ) 

    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'southOutside' );
    cPos = get( hC, 'position' );
    cPos( 2 ) = cPos( 2 ) - .07;
    cPos( 4 ) = .5 * cPos( 4 );
    set( hC, 'position', cPos )
    set( gca, 'position', axPos )


    % (PC1, PC2) angle
    set( gcf, 'currentAxes', ax( 2, 1 ) )
    scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, anglPC, ...
              'filled'  )
    %scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
    plot3( x( 1, idxTPlt( 1 ) : idxTPlt( 2 ) ), ...
           x( 2, idxTPlt( 1 ) : idxTPlt( 2 ) ), ...
           x( 3, idxTPlt( 1 ) : idxTPlt( 2 ) ), 'k-', ...
           'lineWidth', 2 )
    title( '(b) (PC_1, PC_2) angle on L63 attractor' )
    axis off
    view( 90, 0 )
    set( gca, 'cLim', [ -pi pi  ] )
    colormap( gca, jet )
    %set( gcf, 'currentAxes', ax( 2, 1 ) )
    %axis off

    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'southOutside' );
    cPos = get( hC, 'position' );
    cPos( 2 ) = cPos( 2 ) - .07;
    cPos( 4 ) = .5 * cPos( 4 );
    set( hC, 'position', cPos )
    set( gca, 'position', axPos )


    % (PC1,PC2) projection
    set( gcf, 'currentAxes', ax( 3, 1 ) )
    plot( zPC( idxTPlt( 1 ) : idxTPlt( 2 ), 1 ), ...
          zPC( idxTPlt( 1 ) : idxTPlt( 2 ), 2 ), ...
          'b-', 'linewidth', 1.5 ) 
    %scatter( zPC( iT, 1 ), zPC( iT, 2 ), 70, 'r', 'filled' ) 
    xlim( [ -2 2 ] )
    ylim( [ -2 2 ] )
    grid on
    title( '(c) (PC_1, PC_2) projection' )
    xlabel( 'PC_1' )
    ylabel( 'PC_2' )

    
    % PC1 time series
    set( gcf, 'currentAxes', ax( 4, 1 ) )
    plot( t, zPC( idxTPlt( 1 ) : idxTPlt( 2 ), 1 ), 'b-', ...
          'linewidth', 1.5 )
    %scatter( t( iFrame ), zPC( iT, 1 ), 70, 'r', 'filled' ) 
    grid on
    xlim( [ t( 1 ) t( end ) ] )
    ylim( [ -2 2 ] )
    title( '(d) PC_1 time series' )
    xlabel( 't' )
    %ylabel( 'PC_1' )

    % phi1 scatterplot
    set( gcf, 'currentAxes', ax( 1, 2 ) )
    scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, z( :, 1 ), ...
              'filled'  )
    %scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
    %plot3( x( 1, idxTPlt( 1 ) : iT ), x( 2, idxTPlt( 1 ) : iT ), ...
    %       x( 3, idxTPlt( 1 ) : iT ), 'k-', ...
    %        'lineWidth', 2 )
    title( sprintf( '(e) \\phi_{%i} on L63 attractor', idxPhiPlt( 1 ) - 1 ) )
    axis off
    view( 0, 0 )
    set( gca, 'cLim', [ -1 1 ] * a( 1 ) )

    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'southOutside' );
    cPos = get( hC, 'position' );
    cPos( 2 ) = cPos( 2 ) - .07;
    cPos( 4 ) = .5 * cPos( 4 );
    set( hC, 'position', cPos )
    set( gca, 'position', axPos )



    % (phi1,phi2) phase angle
    set( gcf, 'currentAxes', ax( 2, 2 ) )
    scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, angl, 'filled'  )
    %scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
    %plot3( x( 1, idxTPlt( 1 ) : iT ), x( 2, idxTPlt( 1 ) : iT ), ...
    %       x( 3, idxTPlt( 1 ) : iT ), 'm-', ...
    %        'lineWidth', 3 )
    title( sprintf( [ '(f) (\\phi_{%i}, \\phi_{%i}) angle on L63 ' ...
                      'attractor' ], ...
                    idxPhiPlt( 1 ) - 1, idxPhiPlt( 2 ) - 1 ) )
    axis off
    view( 90, 0 )
    set( gca, 'cLim', [ -pi pi  ] )
    colormap( gca, jet )

    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'southOutside' );
    cPos = get( hC, 'position' );
    cPos( 2 ) = cPos( 2 ) - .07;
    cPos( 4 ) = .5 * cPos( 4 );
    set( hC, 'position', cPos )
    set( gca, 'position', axPos )


    % (phi1,phi2) projection
    set( gcf, 'currentAxes', ax( 3, 2 ) )
    plot( z( idxTPlt( 1 ) : idxTPlt( 2 ), 1 ), ...
          z( idxTPlt( 1 ) : idxTPlt( 2 ), 2 ), ...
          'b-', 'linewidth', 1.5 ) 
    %scatter( z( iT, 1 ), z( iT, 2 ), 50, 'r', 'filled' ) 
    xlim( [ -1.5 1.5 ] )
    ylim( [ -1.5 1.5 ] )
    grid on
    title( sprintf( '(g) (\\phi_{%i}, \\phi_{%i}) projection', ...
           idxPhiPlt( 1 ) - 1, idxPhiPlt( 2 ) - 1 ) )
    xlabel( sprintf( '\\phi_{%i}', idxPhiPlt( 1 ) - 1 ) )
    ylabel( sprintf( '\\phi_{%i}', idxPhiPlt( 2 ) - 1 ) )

    % phi1 time series
    set( gcf, 'currentAxes', ax( 4, 2 ) )
    plot( t, z( idxTPlt( 1 ) : idxTPlt( 2 ), 2 ), 'b-', ...
          'linewidth', 1.5 )
    %scatter( t( iFrame ), z( iT, 2 ), 50, 'r', 'filled' ) 
    grid on
    xlim( [ t( 1 ) t( end ) ] )
    ylim( [ -1.5 1.5 ] )
    title( sprintf( '(h) \\phi_{%i} time series', idxPhiPlt( 1 ) - 1 ) )
    xlabel( 't' )


    % Print figure
    if ifPrintFig
        figFile = fullfile( figDir, 'figPhiCoherence.png' );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end




%% MAKE EIGENFUNCTION MOVIE
if ifMovieCoherence
    
    % Retrieve source data and NLSA eigenfunctions
    x = getData( model.srcComponent );
    x = x( :, 1 + nShiftTakens : nSE + nShiftTakens );
    [ phi, ~, lambda ] = getDiffusionEigenfunctions( model );

    % Construct coherent observables based on phi's
    z = phi( :, idxPhiPlt ) .* signPhiPlt / sqrt( 2 );
    a = max( abs( z ), [], 1 ); % z amplitude
    angl = angle( complex( z( :, 1 ), z( :, 2 ) ) ); % z phase

    % Construct coherent observables based on PCs
    zPC = PCA.v( 1 + nShiftTakens : nSE + nShiftTakens, [ 1 2 ] ) ...
        .* signPC / sqrt( 2 );
    anglPC = angle( complex( zPC( :, 1 ), zPC( :, 2 ) ) ); % z phase
    aPC = max( abs( zPC ), [], 1 ); % PC amplitude 

    % Determine number of movie frames; assign timestamps
    nFrame = idxTPlt( 2 ) - idxTPlt( 1 ) + 1;
    t = ( 0 : nFrame - 1 ) * In.dt;  

    % Set up figure and axes 
    Fig.nTileX     = 4;
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
    movieFile = 'movieCoherence.mp4';
    movieFile = fullfile( figDir, movieFile );
    writerObj = VideoWriter( movieFile, 'MPEG-4' );
    writerObj.FrameRate = 20;
    writerObj.Quality = 100;
    open( writerObj );


    % Loop over the frames
    for iFrame = 1 : nFrame

        iT = idxTPlt( 1 ) + iFrame - 1;

        % Scatterplot of PC1
        set( gcf, 'currentAxes', ax( 1, 1 ) )
        scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, zPC( :, 1 ), ...
                  'filled'  )
        scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
        plot3( x( 1, idxTPlt( 1 ) : iT ), x( 2, idxTPlt( 1 ) : iT ), ...
               x( 3, idxTPlt( 1 ) : iT ), 'k-', ...
                'lineWidth', 1.5 )
                  
        title( '(a) PC1 on L63 attracror' )
        axis off
        view( 0, 0 )
        set( gca, 'cLim', [ -1 1 ] * aPC( 1 ) ) 

        axPos = get( gca, 'position' );
        hC = colorbar( 'location', 'southOutside' );
        cPos = get( hC, 'position' );
        cPos( 2 ) = cPos( 2 ) - .07;
        cPos( 4 ) = .5 * cPos( 4 );
        set( hC, 'position', cPos )
        set( gca, 'position', axPos )


        % (PC1, PC2) angle
        set( gcf, 'currentAxes', ax( 2, 1 ) )
        scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, anglPC, ...
                  'filled'  )
        scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
        plot3( x( 1, idxTPlt( 1 ) : iT ), x( 2, idxTPlt( 1 ) : iT ), ...
               x( 3, idxTPlt( 1 ) : iT ), 'm-', ...
                'lineWidth', 3 )
        title( '(b) (PC_1, PC_2) angle on L63 attractor' )
        axis off
        view( 90, 0 )
        set( gca, 'cLim', [ -pi pi  ] )
        colormap( gca, jet )
        %set( gcf, 'currentAxes', ax( 2, 1 ) )
        %axis off

        axPos = get( gca, 'position' );
        hC = colorbar( 'location', 'southOutside' );
        cPos = get( hC, 'position' );
        cPos( 2 ) = cPos( 2 ) - .07;
        cPos( 4 ) = .5 * cPos( 4 );
        set( hC, 'position', cPos )
        set( gca, 'position', axPos )


        % (PC1,PC2) projection
        set( gcf, 'currentAxes', ax( 3, 1 ) )
        plot( zPC( idxTPlt( 1 ) : iT, 1 ), zPC( idxTPlt( 1 ) : iT, 2 ), ...
              'b-', 'linewidth', 1.5 ) 
        scatter( zPC( iT, 1 ), zPC( iT, 2 ), 70, 'r', 'filled' ) 
        xlim( [ -2 2 ] )
        ylim( [ -2 2 ] )
        grid on
        title( '(c) (PC_1, PC_2) projection' )
        xlabel( 'PC_1' )
        ylabel( 'PC_2' )

        
        % PC1 time series
        set( gcf, 'currentAxes', ax( 4, 1 ) )
        plot( t( 1 : iFrame ), zPC( idxTPlt( 1 ) : iT, 1 ), 'b-', ...
              'linewidth', 1.5 )
        scatter( t( iFrame ), zPC( iT, 1 ), 70, 'r', 'filled' ) 
        grid on
        xlim( [ t( 1 ) t( end ) ] )
        ylim( [ -2 2 ] )
        title( '(d) PC_1 time series' )
        xlabel( 't' )
        %ylabel( 'PC_1' )

        % phi1 scatterplot
        set( gcf, 'currentAxes', ax( 1, 2 ) )
        scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, z( :, 1 ), ...
                  'filled'  )
        scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
        plot3( x( 1, idxTPlt( 1 ) : iT ), x( 2, idxTPlt( 1 ) : iT ), ...
               x( 3, idxTPlt( 1 ) : iT ), 'k-', ...
                'lineWidth', 2 )
        title( sprintf( '(e) \\phi_{%i} on L63 attractor', idxPhiPlt( 1 ) - 1 ) )
        axis off
        view( 0, 0 )
        set( gca, 'cLim', [ -1 1 ] * a( 1 ) )

        axPos = get( gca, 'position' );
        hC = colorbar( 'location', 'southOutside' );
        cPos = get( hC, 'position' );
        cPos( 2 ) = cPos( 2 ) - .07;
        cPos( 4 ) = .5 * cPos( 4 );
        set( hC, 'position', cPos )
        set( gca, 'position', axPos )



        % (phi1,phi2) phase angle
        set( gcf, 'currentAxes', ax( 2, 2 ) )
        scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, angl, 'filled'  )
        scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
        plot3( x( 1, idxTPlt( 1 ) : iT ), x( 2, idxTPlt( 1 ) : iT ), ...
               x( 3, idxTPlt( 1 ) : iT ), 'm-', ...
                'lineWidth', 3 )
        title( sprintf( [ '(f) (\\phi_{%i}, \\phi_{%i}) angle on L63 ' ...
                          'attractor' ], ...
                        idxPhiPlt( 1 ) - 1, idxPhiPlt( 2 ) - 1 ) )
        axis off
        view( 90, 0 )
        set( gca, 'cLim', [ -pi pi  ] )
        colormap( gca, jet )

        axPos = get( gca, 'position' );
        hC = colorbar( 'location', 'southOutside' );
        cPos = get( hC, 'position' );
        cPos( 2 ) = cPos( 2 ) - .07;
        cPos( 4 ) = .5 * cPos( 4 );
        set( hC, 'position', cPos )
        set( gca, 'position', axPos )


        % (phi1,phi2) projection
        set( gcf, 'currentAxes', ax( 3, 2 ) )
        plot( z( idxTPlt( 1 ) : iT, 1 ), z( idxTPlt( 1 ) : iT, 2 ), ...
              'b-', 'linewidth', 1.5 ) 
        scatter( z( iT, 1 ), z( iT, 2 ), 50, 'r', 'filled' ) 
        xlim( [ -1.5 1.5 ] )
        ylim( [ -1.5 1.5 ] )
        grid on
        title( sprintf( '(g) (\\phi_{%i}, \\phi_{%i}) projection', ...
               idxPhiPlt( 1 ) - 1, idxPhiPlt( 2 ) - 1 ) )
        xlabel( sprintf( '\\phi_{%i}', idxPhiPlt( 1 ) - 1 ) )
        ylabel( sprintf( '\\phi_{%i}', idxPhiPlt( 2 ) - 1 ) )

        % phi1 time series
        set( gcf, 'currentAxes', ax( 4, 2 ) )
        plot( t( 1 : iFrame ), z( idxTPlt( 1 ) : iT, 2 ), 'b-', ...
              'linewidth', 1.5 )
        scatter( t( iFrame ), z( iT, 2 ), 50, 'r', 'filled' ) 
        grid on
        xlim( [ t( 1 ) t( end ) ] )
        ylim( [ -1.5 1.5 ] )
        title( sprintf( '(h) \\phi_{%i} time series', idxPhiPlt( 1 ) - 1 ) )
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

        
    
