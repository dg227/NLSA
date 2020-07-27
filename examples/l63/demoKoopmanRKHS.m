% DEMO OF KOOPMAN SPECTRAL ANALYSIS FOR THE LORENZ 63 (L63) SYSTEM USING 
% REPRODUCING KERNEL HILBERT SPACE (RKHS) COMPACTIFICATION
% 
% This script computes approximate eigenvalues and eigenfunctions of the 
% generator of the Koopman group on the L2 space associated with the SRB 
% measure of the L63 system. 
%
% 
% Four test cases are provided, illustrating the behavior of the eigenfunctions
% in datasets of either 6400 or 64000 samples, and 0 or 800 delays, all at a
% sampling interval of 0.01 natural time units. The test cases are selected by
% a character variable experiment, which can take the following values:
%
% '6.4k_dt0.01_nEL0':   6400 samples, sampling interval 0.01, 0 delays 
% '6.4k_dt0.01_nEL800': 6400 samples, sampling interval 0.01, 800 delays 
% '64k_dt0.01_nEL0':    64000 samples, sampling interval 0.01, no delays 
% '64k_dt0.01_nEL800':  64000 samples, sampling interval 0.01, 800 delays
%
% In addition, there is an option to run principal component analysis (PCA) on
% the data for comparison.
%
% The kernel employed is a variable-bandwidth Gaussian kernel, normalized to a
% symmetric Markov kernel. This requires a kernel density estimation step to
% compute the bandwidth functions, followed by a normalization step to form the
% kernel matrix. See pseudocode in Appendix B of reference [1] below for 
% further details. 
%
% Approximate running times on Matlab R2018b running on Intel(R) Core(TM) 
% i9-8950HK CPU @ 2.90GHz are as follows:
%
% '6.4k_dt0.01_nEL0':   2 minutes   
% '6.4k_dt0.01_nEL800': 3 minutes
%
% References:
% 
% [1] S. Das, D. Giannakis, J. Slawinska (2018), "Reproducing kernel Hilbert
%     space compactification of unitary evolution groups", 
%     https://arxiv.org/abs/1808.01515.
%
% Modified 2020/07/22

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
%experiment = '6.4k_dt0.01_nEL0'; 
%experiment = '6.4k_dt0.01_nEL800'; 
%experiment = '64k_dt0.01_nEL0'; 
experiment = '64k_dt0.01_nEL100'; 
%experiment = '64k_dt0.01_nEL200'; 
%experiment = '64k_dt0.01_nEL400'; 
%experiment = '64k_dt0.01_nEL800'; 

ifSourceData     = false; % generate source data
ifTestData       = false;  % generate test data for forescasting
ifNLSA           = false; % run NLSA (kernel eigenfunctions)
ifKoopman        = true; % compute Koopman eigenfunctions
ifOse            = false;  % do out-of-sample extension (for forecasting)
ifPCA            = false; % run PCA (for comparison with NLSA)
ifPlotZ          = true; % plot Koopman eigenfunctions
ifPlotCoherence  = false; % figure illustrating coherence of NLSA eigenfunctions
ifMovieCoherence = false; % make eigenfunction movie illustrating coherence
ifPrintFig       = false; % print figures to file

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% GLOBAL PARAMETERS
% idxZPlt:     Eigenfunctions to plot
% nShiftPlt:   Temporal shift applied to eigenfunctions to illustrate action
%              of Koopman operator
% idxTPlt:     Time interval to plot
% figDir:      Output directory for plots
% markerSize:  For eigenfunction scatterplots
% signPC:      Sign multiplication factors for principal components
% signZPlt:    Sign multiplication factors for plotted eigenfunctions 

switch experiment

case '6.4k_dt0.01_nEL0'
    idxZPlt  = [ 1 3 5 7  ];     
    signZPlt = [ 1 -1 ];   
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov times
    markerSize = 7;         
    signPC     = [ -1 1 ];

case '6.4k_dt0.01_nEL800'
    idxZPlt  = [ 9 11 13 15 ];     
    signZPlt = [ 1 -1 ];   
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTPlt    = [ 2001 3001 ]; % approx 10 Lyapunov times
    markerSize = 7;         
    signPC     = [ -1 1 ];

case '64k_dt0.01_nEL0'
    idxZPlt  = [ 1 3 5 7 ];     
    signZPlt = [ 1 -1 ];   
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov times
    markerSize = 3;         
    signPC     = [ -1 1 ];

case '64k_dt0.01_nEL100'
    idxZPlt  = [ 1 3 5 7 ];     
    signZPlt = [ 1 -1 ];   
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov times
    markerSize = 3;         
    signPC     = [ -1 1 ];

case '64k_dt0.01_nEL200'
    idxZPlt  = [ 1 3 5 7 ];     
    signZPlt = [ 1 -1 ];   
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov times
    markerSize = 3;         
    signPC     = [ -1 1 ];

case '64k_dt0.01_nEL400'
    idxZPlt  = [ 1 12 14 ];     
    signZPlt = [ 1 -1 ];   
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov times
    markerSize = 3;         
    signPC     = [ -1 1 ];

case '64k_dt0.01_nEL800'
    idxZPlt  = [ 1 3 43 190 ]; 49;    
    signZPlt = [ 1 -1 ];   
    nShiftPlt  = [ 0 100 300 ]; % approx [ 0 1 3 ] Lyapunov times
    idxTPlt    = [ 2001 3001 ]; % approx 10 Lyapunov times
    markerSize = 3;         
    signPC     = [ -1 1 ];

otherwise
    error( 'Invalid experiment.' )
end


% Figure/movie directory
figDir = fullfile( pwd, 'figs', experiment );
if ~isdir( figDir )
    mkdir( figDir )
end

%% EXTRACT SOURCE DATA
if ifSourceData
    disp( 'Generating source data...' ); t = tic;
    demoKoopmanRKHS_data( experiment ) 
    toc( t )
end

%% EXTRACT TEST DATA
if ifTestData
    disp( 'Generating test data...' ); t = tic;
    demoKoopmanRKHS_test_data( experiment ) 
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
[ model, In ] = demoKoopmanRKHS_nlsaModel( experiment ); 
toc( t )

nSE          = getNTotalSample( model.embComponent );
nSB          = getNXB( model.embComponent );
nEL          = getEmbeddingWindow( model.embComponent ) - 1;
nShiftTakens = round( nEL / 2 );

% Create parallel pool if running NLSA and the NLSA model has been set up
% with parallel workers. This part can be commented out if no parts of the
% NLSA code utilizing parallel workers are being executed. 
%
% In.nParE is the number of parallel workers for delay-embedded distances
% In.nParNN is the number of parallel workers for nearest neighbor search
if ifNLSA
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

%% COMPUTE EIGENFUNCTIONS OF KOOPMAN GENERATOR
if ifKoopman
    disp( 'Koopman eigenfunctions...' ); t = tic;
    computeKoopmanEigenfunctions( model )
    toc( t )
end

%% DO OUT-OF-SAMPLE EXTENSIONS
if ifOse
    
    disp( 'Takens delay embedding for out-of-sample data...' ); t = tic;
    computeOutDelayEmbedding( model )
    toc( t )

    fprintf( 'OSE pairwise distances for density data... %i/%i\n', iProc, ...
        nProc ); t = tic;
    computeOseDenPairwiseDistances( model, iProc, nProc )
    toc( t )

    disp( 'OSE density bandwidth normalization...' ); t = tic;
    computeOseDenBandwidthNormalization( model )
    toc( t )

    disp( 'OSE kernel density estimation...' ); t = tic;
    computeOseDensity( model )
    toc( t )

    disp( 'OSE density delay embedding...' ); t = tic; 
    computeOseDensityDelayEmbedding( model );
    toc( t )

    fprintf( 'OSE pairwise distances... %i/%i\n', iProc, nProc ); t = tic; 
    computeOsePairwiseDistances( model, iProc, nProc )
    toc( t )

    disp( 'OSE kernel normalization...' ); t = tic;
    computeOseKernelNormalization( model )

    disp( 'OSE kernel degree...' ); t = tic;
    computeOseKernelDegree( model )
    toc( t )

    disp( 'Nystrom operator...' ); t = tic;
    computeOseDiffusionOperator( model )
    toc( t )

    disp( 'OSE diffusion eigenfunctions...' ); t = tic;
    computeOseDiffusionEigenfunctions( model )
    toc( t )
end

%% PERFORM PCA
if ifPCA

    disp( 'Principal component analysis...' ); t = tic;
    x = getData( model.srcComponent );
    [ PCA.u, PCA.s, PCA.v ] = svd( x - mean( x, 2 ), 'econ' );
    PCA.v = PCA.v * sqrt( getNTotalSample( model.srcComponent ) );
    toc( t )
end

%% PLOT EIGENFUNCTIONS
if ifPlotZ
    
    % Retrieve source data and NLSA eigenfunctions. Assign timestamps.
    x = getData( model.srcComponent );
    x = x( :, 1 + nShiftTakens : nSE + nShiftTakens );
    [ zeta, ~, omega ] = getKoopmanEigenfunctions( model );
    omega = imag( omega );
    t = ( 0 : nSE - 1 ) * In.dt;  


    % Set up figure and axes 
    Fig.nTileX     = numel( nShiftPlt ) + 1;
    Fig.nTileY     = numel( idxZPlt );
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
    colormap( jet )

    % EIGENFUNCTION SCATTERPLOTS

    % Loop over the time shifts
    for iShift = 1 : Fig.nTileX - 1

        xPlt = x( :, 1 : end - nShiftPlt( iShift ) );

        % Loop over the eigenfunctions
        for iZ = 1 : Fig.nTileY

            zetaPlt = zeta( 1 + nShiftPlt( iShift ) : end, idxZPlt( iZ ) );
            zetaPlt = real( zetaPlt );

            set( gcf, 'currentAxes', ax( iShift, iZ ) )
            scatter3( xPlt( 1, : ), xPlt( 2, : ), xPlt( 3, : ), markerSize, ...
                      zetaPlt, 'filled'  )
            axis off
            view( 0, 0 )
            set( gca, 'cLim', max( abs( zetaPlt ) ) * [ -1 1 ] )
            
            if iShift == 1
                titleStr = sprintf( '\\zeta_{%i}, \\omega_{%i} = %1.3g', ...
                                    idxZPlt( iZ ) - 1, ...
                                    idxZPlt( iZ ) - 1, ...
                                    omega( idxZPlt( iZ ) ) );
            else
                titleStr = sprintf( 'U^t\\zeta_{%i},   t = %1.2f', ...
                                    idxZPlt( iZ ) - 1, ...
                                    nShiftPlt( iShift ) * In.dt ); 
            end
            title( titleStr )
        end

    end

    % EIGENFUNCTION TIME SERIES PLOTS

    tPlt = t( idxTPlt( 1 ) : idxTPlt( 2 ) );
    tPlt = tPlt - tPlt( 1 ); % set time origin to 1st plotted point

    % Loop over the eigenfunctions
    for iZ = 1 : Fig.nTileY

        zetaPlt = zeta( idxTPlt( 1 ) : idxTPlt( 2 ), idxZPlt( iZ ) );

        set( gcf, 'currentAxes', ax( Fig.nTileX, iZ ) )
        plot( tPlt, real( zetaPlt ), '-' )
        grid on
        xlim( [ tPlt( 1 ) tPlt( end ) ] )
        ylim( [ -3 3 ] )

        if iZ == 1
            title( 'Time series along orbit' )
        end
        if iZ == Fig.nTileY
            xlabel( 't' )
        end
    end

    titleStr = [ sprintf( 'Sampling interval \\Deltat = %1.2f, ', In.dt ) ...
                 sprintf( 'Delay embedding window T = %1.2f', In.dt * nEL ) ]; 
    title( axTitle, titleStr )


    % Print figure
    if ifPrintFig
        figFile = sprintf( 'figZ%s.png', idx2str( idxZPlt, '_' ) );
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
    z = phi( :, idxZPlt ) .* signZPlt / sqrt( 2 );
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
    title( sprintf( '(e) \\phi_{%i} on L63 attractor', idxZPlt( 1 ) - 1 ) )
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
                    idxZPlt( 1 ) - 1, idxZPlt( 2 ) - 1 ) )
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
           idxZPlt( 1 ) - 1, idxZPlt( 2 ) - 1 ) )
    xlabel( sprintf( '\\phi_{%i}', idxZPlt( 1 ) - 1 ) )
    ylabel( sprintf( '\\phi_{%i}', idxZPlt( 2 ) - 1 ) )

    % phi1 time series
    set( gcf, 'currentAxes', ax( 4, 2 ) )
    plot( t, z( idxTPlt( 1 ) : idxTPlt( 2 ), 2 ), 'b-', ...
          'linewidth', 1.5 )
    %scatter( t( iFrame ), z( iT, 2 ), 50, 'r', 'filled' ) 
    grid on
    xlim( [ t( 1 ) t( end ) ] )
    ylim( [ -1.5 1.5 ] )
    title( sprintf( '(h) \\phi_{%i} time series', idxZPlt( 1 ) - 1 ) )
    xlabel( 't' )


    % Print figure
    if ifPrintFig
        figFile = fullfile( figDir, 'figZCoherence.png' );
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
    z = phi( :, idxZPlt ) .* signZPlt / sqrt( 2 );
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
        title( sprintf( '(e) \\phi_{%i} on L63 attractor', idxZPlt( 1 ) - 1 ) )
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
                        idxZPlt( 1 ) - 1, idxZPlt( 2 ) - 1 ) )
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
               idxZPlt( 1 ) - 1, idxZPlt( 2 ) - 1 ) )
        xlabel( sprintf( '\\phi_{%i}', idxZPlt( 1 ) - 1 ) )
        ylabel( sprintf( '\\phi_{%i}', idxZPlt( 2 ) - 1 ) )

        % phi1 time series
        set( gcf, 'currentAxes', ax( 4, 2 ) )
        plot( t( 1 : iFrame ), z( idxTPlt( 1 ) : iT, 2 ), 'b-', ...
              'linewidth', 1.5 )
        scatter( t( iFrame ), z( iT, 2 ), 50, 'r', 'filled' ) 
        grid on
        xlim( [ t( 1 ) t( end ) ] )
        ylim( [ -1.5 1.5 ] )
        title( sprintf( '(h) \\phi_{%i} time series', idxZPlt( 1 ) - 1 ) )
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

        
    
