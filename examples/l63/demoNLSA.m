% DEMO OF NLSA APPLIED TO LORENZ 63 DATA
%
% Modified 2020/06/06

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
experiment = '6.4k_dt0.01_nEL0'; % 6400 samples, sampling interval 0.01, 0 delays 
experiment = '6.4k_dt0.01_nEL100'; % 6400 samples, sampling interval 0.01, 100 delays 
experiment = '6.4k_dt0.01_nEL200'; % 6400 samples, sampling interval 0.01, 200 delays 
ifSourceData   = true; % generate source data
ifNLSA         = true; % run NLSA
ifPlotPhi      = true; % plot eigenfunctions
ifPrintFig     = true; % print figures to file

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% GLOBAL PARAMETERS
% nShiftPlt:   Temporal shift applied to eigenfunctions to illustrate action
%                  of Koopman operator
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

% 6400 samples, sampling interval 0.01, 100 delays
case '6.4k_dt0.01_nEL100'

    idxPhiPlt  = [ 2 3 4 5 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         

% 6400 samples, sampling interval 0.01, 100 delays
case '6.4k_dt0.01_nEL200'

    idxPhiPlt  = [ 2 3 4 5 ];     
    nShiftPlt  = [ 0 100 200 ]; % approx [ 0 1 2 ] Lyapunov timescales
    idxTPlt    = [ 2001 3000 ]; % approx 10 Lyapunov timescales
    markerSize = 7;         


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
        figFile = sprintf( 'figPhi_%s.png', idx2str( idxPhiPlt, '_' ) );
        figFile = fullfile( figDir, figFile );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end




    
