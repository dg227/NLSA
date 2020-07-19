% DEMO OF KOOPMAN SPECTRAL ANALYSIS APPLIED TO VARIABLE-SPEED FLOW ON CIRCLE
%
% a is a parameter in the interval ( 0, 1 ] controling the nonlinerity of the 
% flow.  
% 
% a = 1:     Constant-speed flow
% a = 0:     Singular flow with fixed point at theta = 3 * pi / 2
% 0 < a < 1: Flow is faster for 0 < theta < pi and slower for pi < theta < 2 pi. 
%
% Modified 2020/07/15

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
experiment = 'a0.7'; 

ifSourceData         = false; % generate source data
ifNLSA               = false; % run NLSA (kernel eigenfunctions)
ifKoopman            = true;  % compute Koopman eigenfunctions 
ifPlotZ              = false; % plot generator eigenfunctions
ifPlotRectification  = true;  % show dynamical rectification by generator eig  
ifMovieRectification = true;  % make rectification movie
ifPrintFig           = true;  % print figures to file

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% GLOBAL PARAMETERS
% nShiftPlt:   Temporal shift applied to eigenfunctions to illustrate action
%              of Koopman operator
% idxZPlt:     Eigenfunctions to plot
% idxTPlt:     Time interval to plot
% phaseZ:      Phase factor
% idxZRec:     Eigenfunction in rectification figure/movie
% figDir:      Output directory for plots
% markerSize:  For eigenfunction scatterplots

switch experiment

case 'a0.7'
 
    idxZPlt    = [ 2 4 ];     
    phaseZ     = [ 1 ]; 
    nShiftPlt  = [ 0 500 1000 ]; 
    idxTPlt    = [ 1 2001 ]; 
    idxZRec    = 2;
    markerSize = 7;         

otherwise

    'Invalid experiment.'

end

% Figure directory
figDir = fullfile( pwd, 'figs', experiment );
if ~isdir( figDir )
    mkdir( figDir )
end

%% EXTRACT SOURCE DATA
if ifSourceData
    disp( 'Generating source data...' ); t = tic;
    demoKoopman_data( experiment ) 
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
[ model, In ] = demoKoopman_nlsaModel( experiment ); 
toc( t )

nSE          = getNTotalSample( model.embComponent );
nSB          = getNXB( model.embComponent );
nEL          = getEmbeddingWindow( model.embComponent ) - 1;
nShiftTakens = round( nEL / 2 );

% Create parallel pool if required
% In.nParE is the number of parallel workers for delay-embedded distances
% In.nParNN is the number of parallel workers for nearest neighbor search
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

    disp( 'Distance normalization for kernel density estimation...' ); t = tic;
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

%% PLOT EIGENFUNCTIONS
if ifPlotZ
    
    % Retrieve source data and generator eigenfunctions. Assign timestamps.
    x = getData( model.srcComponent );
    x = x( :, 1 + nShiftTakens : nSE + nShiftTakens );
    z = getKoopmanEigenfunctions( model );
    t = ( 0 : nSE - 1 ) * In.Res.dt;  

    % Retrieve eigenperiods and Dirichlet energies
    T = abs( getKoopmanEigenperiods( model ) ); 
    E = getEnergies( model.koopmanOp );

    % Set up figure and axes 
    Fig.nTileX     = numel( nShiftPlt ) + 1;
    Fig.nTileY     = numel( idxZPlt );
    Fig.units      = 'inches';
    Fig.figWidth   = 15 / 4 * Fig.nTileX; 
    Fig.deltaX     = .8;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = 0.7;
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
        for iZ = 1 : Fig.nTileY

            [ ~, iMax ] = max( x( 1, : ) );
            phaseFact = z( iMax, idxZPlt( iZ ) );
            phaseFact = phaseFact / abs( phaseFact );
            zPlt = z( 1 + nShiftPlt( iShift ) : end, idxZPlt( iZ ) ) ...
                 / phaseFact;

            set( gcf, 'currentAxes', ax( iShift, iZ ) )
            scatter( xPlt( 1, : ), xPlt( 2, : ), markerSize, real( zPlt ), ...
                     'filled'  )
            colormap( redblue )
            set( gca, 'cLim', max( abs( zPlt ) ) * [ -1 1 ], ...
                      'color', [ 1 1 1 ] * .3 )
            xlim( [ -1.2 1.2 ] )
            ylim( [ -1.2 1.2 ] )
            if iZ ~= Fig.nTileY
                set( gca, 'xTickLabel', [] )
            end

            if iShift == 1
                axPos = get( gca, 'position' );
                hC = colorbar( 'location', 'westOutside' );
                cPos = get( hC, 'position' );
                cPos( 1 ) = cPos( 1 ) - .04;
                cPos( 3 ) = .5 * cPos( 3 );
                set( hC, 'position', cPos )
                set( gca, 'position', axPos )
            end
            
            set( gca, 'yTickLabel', [] )
            if nShiftPlt( iShift ) == 0 
                titleStr = sprintf( 'z_{%i}, T = %1.3g, E = %1.3g', ...
                                    idxZPlt( iZ ) - 1, T( idxZPlt( iZ ) ), ...
                                    E( idxZPlt( iZ ) ) );

            else
                titleStr = sprintf( 'U^tz_{%i},   t = %1.2f', ...
                                    idxZPlt( iZ ) - 1, ...
                                    nShiftPlt( iShift ) * In.Res.dt ); 
            end
            title( titleStr )
        end

    end

    % EIGENFUNCTION TIME SERIES PLOTS

    tPlt = t( idxTPlt( 1 ) : idxTPlt( 2 ) );
    tPlt = tPlt - tPlt( 1 ); % set time origin to 1st plotted point

    % Loop over the eigenfunctions
    for iZ = 1 : Fig.nTileY

        zPlt = z( idxTPlt( 1 ) : idxTPlt( 2 ), idxZPlt( iZ ) );

        set( gcf, 'currentAxes', ax( Fig.nTileX, iZ ) )
        plot( tPlt, real( zPlt ), '-' )
        grid on
        xlim( [ tPlt( 1 ) tPlt( end ) ] )
        ylim( [ -1.2 1.2 ] )

        if iZ == 1
            title( 'Time series along orbit' )
        end
        if iZ == Fig.nTileY
            xlabel( 't' )
        end
    end

    %titleStr = [ sprintf( 'Sampling interval \\Deltat = %1.2f, ', In.dt ) ...
    %             sprintf( 'Delay embedding window T = %1.2f', In.dt * nEL ) ]; 

    titleStr = sprintf( [ 'Generator eigenfunctions (real part) for ' ...
                          'variable-speed circle rotation, ' ...
                          'f = %1.3g, \\alpha = %1.3g' ], ...
                          In.Res.f, In.Res.a );
    title( axTitle, titleStr )
    % Print figure
    if ifPrintFig
        figFile = sprintf( 'figZ%s.png', idx2str( idxZPlt, '_' ) );
        figFile = fullfile( figDir, figFile );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end

%% DYNAMICAL RECTIFICATION FIGURE
if ifPlotRectification
    
    % Retrieve source data and generator eigenfunctions. Assign timestamps.
    x = getData( model.srcComponent );
    x = x( :, 1 + nShiftTakens : nSE + nShiftTakens );
    z = getKoopmanEigenfunctions( model );
    t = ( 0 : nSE - 1 ) * In.Res.dt;  
    tPlt = t( idxTPlt( 1 ) : idxTPlt( 2 ) );
    tPlt = tPlt - tPlt( 1 ); % set time origin to 1st plotted point

    [ ~, iMax ] = max( x( 1, : ) );
    phaseFact = z( iMax, idxZRec );
    phaseFact = phaseFact / abs( phaseFact );
    zPlt = z( :, idxZRec ) / phaseFact;

    % Set up figure and axes 
    Fig.nTileX     = 2;
    Fig.nTileY     = 2;
    Fig.units      = 'inches';
    Fig.figWidth   = 15 / 4 * Fig.nTileX; 
    Fig.deltaX     = .8;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = 0.7;
    Fig.gapT       = 0; 
    Fig.aspectR    = 3 / 4;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );


    % Generator eigenfunction on original (unrectified) state space
    set( gcf, 'currentAxes', ax( 1, 1 ) )
    scatter( x( 1, : ), x( 2, : ), markerSize, real( zPlt ), 'filled'  )
    colormap( redblue )
    set( gca, 'cLim', max( abs( zPlt ) ) * [ -1 1 ], ...
              'color', [ 1 1 1 ] * .3, ...
              'xTickLabel', [], 'yTickLabel', [] )
    xlim( [ -1.2 1.2 ] )
    ylim( [ -1.2 1.2 ] )
    title( sprintf( 'Re(z_{%i}) on original state space' ,idxZRec - 1 ) )
    
    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'westOutside' );
    cPos = get( hC, 'position' );
    cPos( 1 ) = cPos( 1 ) - .08;
    cPos( 3 ) = .5 * cPos( 3 );
    set( hC, 'position', cPos )
    set( gca, 'position', axPos )
            
    % Time series of generator eigenfunction
    set( gcf, 'currentAxes', ax( 2, 1 ) )
    plot( tPlt, real( zPlt( idxTPlt( 1 ) : idxTPlt( 2 ) ) ), '-' )
    grid on
    xlim( [ tPlt( 1 ) tPlt( end ) ] )
    ylim( [ -1.2 1.2 ] )
    title( 'Time series along orbit of original system' )

    % x1 observable on rectified state space
    set( gcf, 'currentAxes', ax( 1, 2 ) )
    scatter( real( zPlt ), imag( zPlt ), markerSize, x( 1, : ), 'filled'  )
    colormap( redblue )
    set( gca, 'cLim', [ -1 1 ], ...
              'color', [ 1 1 1 ] * .3, ...
              'xTickLabel', [], 'yTickLabel', [] )
    xlim( [ -1.2 1.2 ] )
    ylim( [ -1.2 1.2 ] )
    title( 'x_1 on rectified state space' )
    
    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'westOutside' );
    cPos = get( hC, 'position' );
    cPos( 1 ) = cPos( 1 ) - .08;
    cPos( 3 ) = .5 * cPos( 3 );
    set( hC, 'position', cPos )
    set( gca, 'position', axPos )
 
    % Time series of x1 observable
    set( gcf, 'currentAxes', ax( 2, 2 ) )
    plot( tPlt, x( 1, idxTPlt( 1 ) : idxTPlt( 2 ) ), '-' )
    grid on
    xlim( [ tPlt( 1 ) tPlt( end ) ] )
    ylim( [ -1.2 1.2 ] )
    xlabel( 't' ) 
    title( 'Time series along orbit of original system' )
    %titleStr = [ sprintf( 'Sampling interval \\Deltat = %1.2f, ', In.dt ) ...
    %             sprintf( 'Delay embedding window T = %1.2f', In.dt * nEL ) ]; 

    %titleStr = sprintf( [ 'Generator eigenfunctions (real part) for ' ...
    %                      'variable-speed circle rotation, ' ...
    %                      'f = %1.3g, \\alpha = %1.3g' ], ...
    %                      In.Res.f, In.Res.a );
    %title( axTitle, titleStr )
    % Print figure
    if ifPrintFig
        figFile = 'figRectification.png';
        figFile = fullfile( figDir, figFile );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end



%% MAKE MOVIE ILLUSTRATING RECTIFICATION
if ifMovieRectification
    
    % Retrieve source data and generator eigenfunctions. Assign timestamps.
    x = getData( model.srcComponent );
    x = x( :, 1 + nShiftTakens : nSE + nShiftTakens );
    z = getKoopmanEigenfunctions( model );
    t = ( 0 : nSE - 1 ) * In.Res.dt;  
    tPlt = t( idxTPlt( 1 ) : idxTPlt( 2 ) );
    tPlt = tPlt - tPlt( 1 ); % set time origin to 1st plotted point
    
    [ ~, iMax ] = max( x( 1, : ) );
    phaseFact = z( iMax, idxZRec );
    phaseFact = phaseFact / abs( phaseFact );
    zPlt = z( :, idxZRec ) / phaseFact;

    % Set up figure and axes 
    Fig.nTileX     = 2;
    Fig.nTileY     = 2;
    Fig.units      = 'pixels';
    Fig.figWidth   = 500; 
    Fig.deltaX     = 50;
    Fig.deltaX2    = 20;
    Fig.deltaY     = 50;
    Fig.deltaY2    = 30;
    Fig.gapX       = 50;
    Fig.gapY       = 70;
    Fig.gapT       = 30; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Set up videowriter
    movieFile = 'movieRectification.mp4';
    movieFile = fullfile( figDir, movieFile );
    writerObj = VideoWriter( movieFile, 'MPEG-4' );
    writerObj.FrameRate = 50;
    writerObj.Quality = 100;
    open( writerObj );

    % Determine number of movie frames; assign timestamps
    nFrame = idxTPlt( 2 ) - idxTPlt( 1 ) + 1;
    t = ( 0 : nFrame - 1 ) * In.Res.dt;  

    % Loop over the frames
    for iFrame = 1 : nFrame

        iT = idxTPlt( 1 ) + iFrame - 1;

        % Generator eigenfunction on original (unrectified) state space
        set( gcf, 'currentAxes', ax( 1, 1 ) )
        scatter( x( 1, : ), x( 2, : ), markerSize, real( zPlt ), 'filled'  )
        scatter( x( 1, iT ), x( 2, iT ), 70, 'g', 'filled' ) 
        colormap( redblue )
        set( gca, 'cLim', max( abs( zPlt ) ) * [ -1 1 ], ...
                  'color', [ 1 1 1 ] * .3, ...
                  'xTickLabel', [], 'yTickLabel', [] )
        xlim( [ -1.2 1.2 ] )
        ylim( [ -1.2 1.2 ] )
        title( sprintf( 'Re(z_{%i}) on original state space' ,idxZRec - 1 ) )

        axPos = get( gca, 'position' );
        hC = colorbar( 'location', 'westOutside' );
        cPos = get( hC, 'position' );
        cPos( 1 ) = cPos( 1 ) - .07;
        cPos( 3 ) = .5 * cPos( 3 );
        set( hC, 'position', cPos )
        set( gca, 'position', axPos )
     
        % Time series of generator eigenfunction
        set( gcf, 'currentAxes', ax( 2, 1 ) )
        plot( tPlt( 1 : iFrame ), real( zPlt( idxTPlt( 1 ) : iT ) ), '-', ...
              'linewidth', 1 )
        scatter( tPlt( iFrame ), real( zPlt( iT ) ), 50, 'g', 'filled' )
        grid on
        xlim( [ tPlt( 1 ) tPlt( end ) ] )
        ylim( [ -1.2 1.2 ] )
        title( 'Time series along orbit of original system' )

        % x1 observable on rectified state space
        set( gcf, 'currentAxes', ax( 1, 2 ) )
        scatter( real( zPlt ), imag( zPlt ), markerSize, x( 1, : ), 'filled'  )
        scatter( real( zPlt( iT ) ), imag( zPlt( iT )), 70, 'g', 'filled' ) 
        colormap( redblue )
        set( gca, 'cLim', [ -1 1 ], ...
                  'color', [ 1 1 1 ] * .3, ...
                  'xTickLabel', [], 'yTickLabel', [] )
        xlim( [ -1.2 1.2 ] )
        ylim( [ -1.2 1.2 ] )
        title( 'x_1 on rectified state space' )
        
        axPos = get( gca, 'position' );
        hC = colorbar( 'location', 'westOutside' );
        cPos = get( hC, 'position' );
        cPos( 1 ) = cPos( 1 ) - .07;
        cPos( 3 ) = .5 * cPos( 3 );
        set( hC, 'position', cPos )
        set( gca, 'position', axPos )

        % Time series of x1 observable
        set( gcf, 'currentAxes', ax( 2, 2 ) )
        plot( tPlt( 1 : iFrame ), x( 1, idxTPlt( 1 ) : iT ), '-', ...
              'linewidth', 1 )
        scatter( tPlt( iFrame ), x( 1, iT ), 50, 'g', 'filled' )
        grid on
        xlim( [ tPlt( 1 ) tPlt( end ) ] )
        ylim( [ -1.2 1.2 ] )
        xlabel( 't' ) 
        title( 'Time series along orbit of original system' )

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

        
    
