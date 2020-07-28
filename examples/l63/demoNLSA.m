% DEMO OF NLSA APPLIED TO LORENZ 63 (L63) DATA
%
% This script demonstrates the identification of coherent observables of the L63
% system using eigenfunctions of kernel integral operators constructed from 
% delay-coordinate mapped data. 
%
% Given a sufficiently long delay embedding window, the leading two 
% eigenfunctions of these operators exhibit an approximately cyclical behavior 
% despite the mixing (chaotic) dynamics of L63, on timescales of order 10 
% Lyapunov times.
%
% In contrast, without delays, the leading eigenfunctions exhibit a chaotic
% behavior of comparable complexity to the state vector components of the L63
% system. 
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
% kernel matrix. See pseudocode in Appendix B of reference [2] below for 
% further details. 
%
% The script calls function demoNLSA_nlsaModel to create an object of
% nlsaModel class, which encodes all aspects and parameters of the calculation,
% such as location of source data, kernel parameters, Koopman operator 
% approximation parameters, etc. 
%
% Results from each stage of the calculation are written on disk. Below is a
% summary of basic commands to access the code output:
%
% lambda = getDiffusionEigenvalues( model ); -- NLSA (kernel) eigenvalues
% phi    = getDiffusionEigenfunctions( model ); -- kernel eigenfunctions
% z     = getKoopmanEigenfunctions( model );   -- Koopman eigenfunctions
% gamma = getKoopmanEigenvalues( model ); -- Koopman eigenvalues  
% T     = getKoopmanEigenperiods( model ); -- Koopman eigenperiods
%
% Approximate running times on Matlab R2018b running on Intel(R) Core(TM) 
% i9-8950HK CPU @ 2.90GHz are as follows:
%
% '6.4k_dt0.01_nEL0':   2 minutes   
% '6.4k_dt0.01_nEL800': 3 minutes
% '64k_dt0.01_nEL0':    45 minutes
% '64k_dt0.01_nEL800':  90 minutes
%
% See function demoNLSA_nlsaModel for additional information on dataset 
% partitioning and parallelization using Matlab's Parallel Computing toolbox.
%
% A figure, figPhiCoherence.png, depicting the extracted coherent observables 
% from the '64k_dt0.01_nEL800' experiment, is included for reference in 
% subdirectory ./figs/64k_dt0.01_nEL800.
%
% References:
% 
% [1] S. Das, D. Giannakis (2019), "Delay-coordinate maps and the spectra of
%     Koopman operators", J. Stat. Phys., 175(6), 1107-1145, 
%     https://doi.org/10.1007/s10955-019-02272-w.
%
% [2] S. Das, D. Giannakis, J. Slawinska (2018), "Reproducing kernel Hilbert
%     space compactification of unitary evolution groups", 
%     https://arxiv.org/abs/1808.01515.
%
% [3] D. Giannakis, A. J. Majda (2012), "Nonlinear Laplacian spectral analysis
%     for time series with intermittency and low-frequency variability", 
%     Proc. Natl. Acad. Sci., 109(7), 2222, 
%     http://dx.doi.org/10.1073/pnas.1118984109.
%
% [4] D. Giannakis (2019), "Data-driven spectral decomposition and forecasting
%     of ergodic dynamical systems", Appl. Comput. Harmon. Anal., 62(2), 
%     338-396, http://dx.doi.org/10.1016/j.acha.2017.09.001.
% 
% [5] D. Giannakis (2020), "Delay-coordinate maps, coherence, and approximate 
%     spectra of evolution operators", https://arxiv.org/abs/2007.02195.  
%
% Modified 2020/07/22

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
%experiment = '6.4k_dt0.01_nEL0'; 
%experiment = '6.4k_dt0.01_nEL800'; 
%experiment = '64k_dt0.01_nEL0'; 
experiment = '64k_dt0.01_nEL800'; 

ifSourceData     = false; % generate source data
ifNLSA           = false; % run NLSA (kernel eigenfunctions)
ifPCA            = false; % run PCA (for comparison with NLSA)
ifPlotPhi        = false; % plot eigenfunctions
ifPlotCoherence  = false; % figure illustrating coherence of NLSA eigenfunctions
ifMovieCoherence = false; % make eigenfunction movie illustrating coherence
ifNLSAPhases     = true;  % identify "lifecycle" phases from eigenfunctions
ifPrintFig       = false; % print figures to file

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% GLOBAL PARAMETERS
% idxPhi:     Eigenfunctions to plot
% nShift:     Temporal shift applied to eigenfunctions to illustrate action
%             of Koopman operator
% idxTLim:    Time interval to plot
% figDir:     Output directory for plots
% markerSize: For eigenfunction scatterplots
% signPC:     Sign multiplication factors for principal components
% signPhi:    Sign multiplication factors for plotted eigenfunctions 

decayFactor  = 4; 

switch experiment

case '6.4k_dt0.01_nEL0'
    idxPhi     = [ 2 3 ];     
    signPhi    = [ 1 -1 ];   
    nShift     = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTLim    = [ 2001 3000 ]; % approx 10 Lyapunov times
    markerSize = 7;         
    signPC     = [ -1 1 ];

case '6.4k_dt0.01_nEL800'
    idxPhi     = [ 2 3 ];     
    signPhi    = [ 1 -1 ];   
    nShift     = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTLim    = [ 2001 3001 ]; % approx 10 Lyapunov times
    markerSize = 7;         
    signPC     = [ -1 1 ];

case '64k_dt0.01_nEL0'
    idxPhi     = [ 2 3 ];     
    signPhi    = [ 1 -1 ];   
    nShift     = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTLim    = [ 2001 3000 ]; % approx 10 Lyapunov times
    markerSize = 3;         
    signPC     = [ -1 1 ];

case '64k_dt0.01_nEL800'
    idxPhi     = [ 2 3 ];     
    signPhi    = [ 1 -1 ];   
    nShift     = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov times
    idxTLim    = [ 2001 3001 ]; % approx 10 Lyapunov times
    markerSize = 3;         
    signPC     = [ -1 1 ];
    nPhase     = 16;
    nSPhase    = 200;
    

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

%% PERFORM PCA
if ifPCA

    disp( 'Principal component analysis...' ); t = tic;
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
    Fig.nTileX     = numel( nShift ) + 1;
    Fig.nTileY     = numel( idxPhi );
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

        xPlt = x( :, 1 : end - nShift( iShift ) );

        % Loop over the eigenfunctions
        for iPhi = 1 : Fig.nTileY

            phiPlt = phi( 1 + nShift( iShift ) : end, idxPhi( iPhi ) );

            set( gcf, 'currentAxes', ax( iShift, iPhi ) )
            scatter3( xPlt( 1, : ), xPlt( 2, : ), xPlt( 3, : ), markerSize, ...
                      phiPlt, 'filled'  )
            axis off
            view( 0, 0 )
            set( gca, 'cLim', max( abs( phiPlt ) ) * [ -1 1 ] )
            
            if iShift == 1
                titleStr = sprintf( '\\phi_{%i}, \\lambda_{%i} = %1.3g', ...
                                    idxPhi( iPhi ) - 1, ...
                                    idxPhi( iPhi ) - 1, ...
                                    lambda( idxPhi( iPhi ) ) );
            else
                titleStr = sprintf( 'U^t\\phi_{%i},   t = %1.2f', ...
                                    idxPhi( iPhi ) - 1, ...
                                    nShift( iShift ) * In.dt ); 
            end
            title( titleStr )
        end

    end

    % EIGENFUNCTION TIME SERIES PLOTS

    tPlt = t( idxTLim( 1 ) : idxTLim( 2 ) );
    tPlt = tPlt - tPlt( 1 ); % set time origin to 1st plotted point

    % Loop over the eigenfunctions
    for iPhi = 1 : Fig.nTileY

        phiPlt = phi( idxTLim( 1 ) : idxTLim( 2 ), idxPhi( iPhi ) );

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
        figFile = sprintf( 'figPhi%s.png', idx2str( idxPhi, '_' ) );
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
    z = phi( :, idxPhi ) .* signPhi / sqrt( 2 );
    a = max( abs( z ), [], 1 ); % z amplitude
    angl = angle( complex( z( :, 1 ), z( :, 2 ) ) ); % z phase

    % Construct coherent observables based on PCs
    zPC = PCA.v( 1 + nShiftTakens : nSE + nShiftTakens, [ 1 2 ] ) ...
        .* signPC / sqrt( 2 );
    anglPC = angle( complex( zPC( :, 1 ), zPC( :, 2 ) ) ); % z phase
    aPC = max( abs( zPC ), [], 1 ); % PC amplitude 

    % Determine number of temporal samples; assign timestamps
    nFrame = idxTLim( 2 ) - idxTLim( 1 ) + 1;
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
    plot3( x( 1, idxTLim( 1 ) : idxTLim( 2 ) ), ...
           x( 2, idxTLim( 1 ) : idxTLim( 2 ) ), ...
           x( 3, idxTLim( 1 ) : idxTLim( 2 ) ), 'k-', ...
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
    plot3( x( 1, idxTLim( 1 ) : idxTLim( 2 ) ), ...
           x( 2, idxTLim( 1 ) : idxTLim( 2 ) ), ...
           x( 3, idxTLim( 1 ) : idxTLim( 2 ) ), 'k-', ...
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
    plot( zPC( idxTLim( 1 ) : idxTLim( 2 ), 1 ), ...
          zPC( idxTLim( 1 ) : idxTLim( 2 ), 2 ), ...
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
    plot( t, zPC( idxTLim( 1 ) : idxTLim( 2 ), 1 ), 'b-', ...
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
    %plot3( x( 1, idxTLim( 1 ) : iT ), x( 2, idxTLim( 1 ) : iT ), ...
    %       x( 3, idxTLim( 1 ) : iT ), 'k-', ...
    %        'lineWidth', 2 )
    title( sprintf( '(e) \\phi_{%i} on L63 attractor', idxPhi( 1 ) - 1 ) )
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
    %plot3( x( 1, idxTLim( 1 ) : iT ), x( 2, idxTLim( 1 ) : iT ), ...
    %       x( 3, idxTLim( 1 ) : iT ), 'm-', ...
    %        'lineWidth', 3 )
    title( sprintf( [ '(f) (\\phi_{%i}, \\phi_{%i}) angle on L63 ' ...
                      'attractor' ], ...
                    idxPhi( 1 ) - 1, idxPhi( 2 ) - 1 ) )
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
    plot( z( idxTLim( 1 ) : idxTLim( 2 ), 1 ), ...
          z( idxTLim( 1 ) : idxTLim( 2 ), 2 ), ...
          'b-', 'linewidth', 1.5 ) 
    %scatter( z( iT, 1 ), z( iT, 2 ), 50, 'r', 'filled' ) 
    xlim( [ -1.5 1.5 ] )
    ylim( [ -1.5 1.5 ] )
    grid on
    title( sprintf( '(g) (\\phi_{%i}, \\phi_{%i}) projection', ...
           idxPhi( 1 ) - 1, idxPhi( 2 ) - 1 ) )
    xlabel( sprintf( '\\phi_{%i}', idxPhi( 1 ) - 1 ) )
    ylabel( sprintf( '\\phi_{%i}', idxPhi( 2 ) - 1 ) )

    % phi1 time series
    set( gcf, 'currentAxes', ax( 4, 2 ) )
    plot( t, z( idxTLim( 1 ) : idxTLim( 2 ), 2 ), 'b-', ...
          'linewidth', 1.5 )
    %scatter( t( iFrame ), z( iT, 2 ), 50, 'r', 'filled' ) 
    grid on
    xlim( [ t( 1 ) t( end ) ] )
    ylim( [ -1.5 1.5 ] )
    title( sprintf( '(h) \\phi_{%i} time series', idxPhi( 1 ) - 1 ) )
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
    z = phi( :, idxPhi ) .* signPhi / sqrt( 2 );
    a = max( abs( z ), [], 1 ); % z amplitude
    angl = angle( complex( z( :, 1 ), z( :, 2 ) ) ); % z phase

    % Construct coherent observables based on PCs
    zPC = PCA.v( 1 + nShiftTakens : nSE + nShiftTakens, [ 1 2 ] ) ...
        .* signPC / sqrt( 2 );
    anglPC = angle( complex( zPC( :, 1 ), zPC( :, 2 ) ) ); % z phase
    aPC = max( abs( zPC ), [], 1 ); % PC amplitude 

    % Determine number of movie frames; assign timestamps
    nFrame = idxTLim( 2 ) - idxTLim( 1 ) + 1;
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

        iT = idxTLim( 1 ) + iFrame - 1;

        % Scatterplot of PC1
        set( gcf, 'currentAxes', ax( 1, 1 ) )
        scatter3( x( 1, : ), x( 2, : ), x( 3, : ), markerSize, zPC( :, 1 ), ...
                  'filled'  )
        scatter3( x( 1, iT ), x( 2, iT ), x( 3, iT ), 70, 'r', 'filled' ) 
        plot3( x( 1, idxTLim( 1 ) : iT ), x( 2, idxTLim( 1 ) : iT ), ...
               x( 3, idxTLim( 1 ) : iT ), 'k-', ...
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
        plot3( x( 1, idxTLim( 1 ) : iT ), x( 2, idxTLim( 1 ) : iT ), ...
               x( 3, idxTLim( 1 ) : iT ), 'm-', ...
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
        plot( zPC( idxTLim( 1 ) : iT, 1 ), zPC( idxTLim( 1 ) : iT, 2 ), ...
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
        plot( t( 1 : iFrame ), zPC( idxTLim( 1 ) : iT, 1 ), 'b-', ...
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
        plot3( x( 1, idxTLim( 1 ) : iT ), x( 2, idxTLim( 1 ) : iT ), ...
               x( 3, idxTLim( 1 ) : iT ), 'k-', ...
                'lineWidth', 2 )
        title( sprintf( '(e) \\phi_{%i} on L63 attractor', idxPhi( 1 ) - 1 ) )
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
        plot3( x( 1, idxTLim( 1 ) : iT ), x( 2, idxTLim( 1 ) : iT ), ...
               x( 3, idxTLim( 1 ) : iT ), 'm-', ...
                'lineWidth', 3 )
        title( sprintf( [ '(f) (\\phi_{%i}, \\phi_{%i}) angle on L63 ' ...
                          'attractor' ], ...
                        idxPhi( 1 ) - 1, idxPhi( 2 ) - 1 ) )
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
        plot( z( idxTLim( 1 ) : iT, 1 ), z( idxTLim( 1 ) : iT, 2 ), ...
              'b-', 'linewidth', 1.5 ) 
        scatter( z( iT, 1 ), z( iT, 2 ), 50, 'r', 'filled' ) 
        xlim( [ -1.5 1.5 ] )
        ylim( [ -1.5 1.5 ] )
        grid on
        title( sprintf( '(g) (\\phi_{%i}, \\phi_{%i}) projection', ...
               idxPhi( 1 ) - 1, idxPhi( 2 ) - 1 ) )
        xlabel( sprintf( '\\phi_{%i}', idxPhi( 1 ) - 1 ) )
        ylabel( sprintf( '\\phi_{%i}', idxPhi( 2 ) - 1 ) )

        % phi1 time series
        set( gcf, 'currentAxes', ax( 4, 2 ) )
        plot( t( 1 : iFrame ), z( idxTLim( 1 ) : iT, 2 ), 'b-', ...
              'linewidth', 1.5 )
        scatter( t( iFrame ), z( iT, 2 ), 50, 'r', 'filled' ) 
        grid on
        xlim( [ t( 1 ) t( end ) ] )
        ylim( [ -1.5 1.5 ] )
        title( sprintf( '(h) \\phi_{%i} time series', idxPhi( 1 ) - 1 ) )
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

        
    
%% COMPUTE AND PLOT L63 "PHASES" BASED ON NLSA EIGENFUNCTIONS
if ifNLSAPhases

    disp( 'NLSA-based L63 phases...' ); t = tic;

    % Construct coherent observables based on phi's
    z = phi( :, idxPhi ) .* signPhi / sqrt( 2 );

    % Compute ENSO phases based on NLSA
    [ selectInd, angles, ~, weights ] = ...
        computeLifecyclePhasesWeighted( z, z( :, 1 ), nPhase, nSPhase, ...
                                        decayFactor );


    % Start and end time indices in data arrays
    iStart  = 1 + nSB + nShiftTakens;
    iEnd    = iStart + nSE - 1;  
    compPhi = computePhaseComposites( model, selectInd, ...
                                      iStart, iEnd, weights );
    compPhi = compPhi{ 1 };

    toc( t )


    % Set up figure and axes 
    Fig.nTileX     = 2;
    Fig.nTileY     = 1;
    Fig.units      = 'inches';
    Fig.figWidth   = 9; 
    Fig.deltaX     = .7;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .65;
    Fig.deltaY2    = .4;
    Fig.gapX       = .70;
    Fig.gapY       = 1;
    Fig.gapT       = 0; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );

    % Plot phases
    set( gcf, 'currentAxes', ax( 1, 1 ) )
    c = distinguishable_colors( nPhase );
    plot( z( :, 1 ), z( :, 2 ), '-', 'color', [ 1 1 1 ] * .7 )

    for iPhase = 1 : nPhase
        plot( z( selectInd{ iPhase }, 1 ), ...
              z( selectInd{ iPhase }, 2 ), ...
              '.', 'markersize', 15, 'color', c( iPhase, : ) )
    end
    grid on
    title( sprintf( '(a) (\\phi_{%i}, \\phi_{%i}) phases', ...
           idxPhi( 1 ) - 1, idxPhi( 2 ) - 1 ) )
    xlabel( sprintf( '\\phi_{%i}', idxPhi( 1 ) - 1 ) )
    ylabel( sprintf( '\\phi_{%i}', idxPhi( 2 ) - 1 ) )
    xlim( [ -1.5 1.5 ] )
    ylim( [ -1.5 1.5 ] )

    % Plot phase composites
    set( gcf, 'currentAxes', ax( 2, 1 ) )
    compPlt = [ compPhi compPhi( :, 1 ) ]; % augment periodically
    plot3( compPlt( 1, : ), compPlt( 2, : ), compPlt( 3, : ), '-', ...
           'color', [ 1 1 1 ] * 0 ) 
    for iPhase = 1 : nPhase
        scatter3( compPhi( 1, iPhase ), compPhi( 2, iPhase ), ...
                  compPhi( 3, iPhase ), 50, c( iPhase, : ), 'filled' )
    end
    view( 50, 20 )
    grid on
    title( '(b) Phase composites of state vector' )
    xlabel( 'x' )
    ylabel( 'y' )
    zlabel( 'z' )

    % Print figure
    if ifPrintFig
        figFile = fullfile( figDir, 'figPhiLifecycle.png' );
        print( fig, figFile, '-dpng', '-r300' ) 
    end

end


%% AUXILIARY FUNCTIONS
%
% Function to compute phase composites from target data of NLSA model
function comp = computePhaseComposites( model, selectInd, iStart, iEnd, ...
                                        weights )
nC = size( model.trgComponent, 1 ); % number of observables to be composited
nPhase = numel( selectInd ); % number of phases       
ifWeights = nargin == 5; % 

comp = cell( 1, nC );

% Loop over the components
for iC = 1 : nC

    % Read data from NLSA model  
    y = getData( model.trgComponent( iC ) );
    y = y( :, iStart : iEnd ); 
        
    nD = size( y, 1 ); % data dimension
    comp{ iC } = zeros( nD, nPhase );

        % Loop over the phases
        for iPhase = 1 : nPhase

            % Compute phase conditional average
            if ifWeights
                comp{ iC }( :, iPhase ) = y * weights{ iPhase };
            else    
                comp{ iC }( :, iPhase ) = ...
                    mean( y( :, selectInd{ iPhase } ), 2 );
            end

        end
    end
end


