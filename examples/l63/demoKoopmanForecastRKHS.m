% DEMO OF FORECASTING OF THE LORENZ 63 (L63) SYSTEM USING REPRODUCING KERNEL 
% HILBERT SPACE (RKHS) COMPACTIFICATION OF THE KOOPMAN GENERATOR
% 
% In this example, we use a skew-adjoint, compact approximation of the 
% generator of the Kooopman group of the L63 system to forecast its state 
% vector components. This approximation is constructed by first computing
% data-driven orthonormal basis functions for a suitable RKHS of observables of
% the system using eigenfunctions of kernel integral operator. The kernel 
% eigenfunctions are then employed to represent the operator as a 
% skew-symmetric matrix, and finally that matrix is conjugated by a diagonal 
% matrix representing a smoothing operator. 
%
% Forecasting is performed by exponentiating the compactified generator, 
% leading to a unitary approximation of the Koopman evolution operator of the
% system. 
% 
% Two test cases are provided, illustrating the behavior of the method in 
% datasets of either 6400 or 64000 samples.
%
% '6.4k_dt0.01_nEL0':   6400 samples, sampling interval 0.01, 0 delays 
% '64k_dt0.01_nEL0':    64000 samples, sampling interval 0.01, no delays 
%
% The kernel employed is a variable-bandwidth Gaussian kernel, normalized to a
% symmetric Markov kernel. This requires a kernel density estimation step to
% compute the bandwidth functions, followed by a normalization step to form the
% kernel matrix. See pseudocode in Appendix B of reference [1] below for 
% further details. 
%
% References:
% 
% [1] S. Das, D. Giannakis, J. Slawinska (2018), "Reproducing kernel Hilbert
%     space compactification of unitary evolution groups", 
%     https://arxiv.org/abs/1808.01515.
%
% Modified 2020/08/07

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
%experiment = '6.4k_dt0.01_nEL0'; 
experiment = '64k_dt0.01_nEL0'; 

ifSourceData       = false; % generate source data
ifTestData         = false; % generate test data for forescasting
ifNLSA             = false; % run NLSA (kernel eigenfunctions)
ifKoopman          = false; % compute Koopman eigenfunctions
ifOse              = false; % do out-of-sample extension (for forecasting)
ifReadForecastData = false;  % read training/test data for forecasting 
ifForecast         = false;  % perform forecasting
ifPlotForecast     = false; % plot representative forecast trajectories
ifPlotError        = false; % plot forecast error
ifPrintFig         = false; % print figures to file

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% GLOBAL PARAMETERS AND VARIABLES
% nL:         Number of eigenfunctions to use for operator approximation
% nT:         Number of timesteps to predict (including 0)
% idxTPlt:    Initial conditions for which to plot forecast trajectories
% tStr:       String to prepend in figure titles       
% figDir:     Output directory for plots

switch experiment

case '6.4k_dt0.01_nEL0'
    nT         = 500;
    nL         = 500;
    idxTPlt    = [ 2101 2201 2301 ]; 
    tStr       = '6.4k';

case '64k_dt0.01_nEL0'
    nT         = 500;
    nL         = 2000;
    idxTPlt    = [ 2101 2201 2301 ]; 
    tStr       = '64k';
otherwise
    error( 'Invalid experiment.' )
end


% Figure/movie directory
figDir = fullfile( pwd, 'figsKoopmanForecastRKHS', experiment );
if ~isdir( figDir )
    mkdir( figDir )
end

%% EXTRACT SOURCE DATA
if ifSourceData
    disp( 'Generating source data...' ); t = tic;
    demoKoopmanForecastRKHS_data( experiment ) 
    toc( t )
end

%% EXTRACT TEST DATA
if ifTestData
    disp( 'Generating test data...' ); t = tic;
    demoKoopmanForecastRKHS_test_data( experiment ) 
    toc( t )
end


%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES
% In and Out are data structures containing the NLSA parameters for the 
% training and test data, respectively.
%
% nSX is the number of covariate samples available for training. 
% 
% idxT1 is the initial time stamp in the covariante data ("origin") where 
% delay embedding is performed. We remove samples 1 to idxT1 from the response
% data so that they always lie in the future of the delay embedding window.
%
% idxPhi are the indices of the diffusion eigenfunctions used for RKHS 
% compactification.
%
% tauRKHS is the RKHS compactification parameter. 

disp( 'Building NLSA model...' ); t = tic;
[ model, In, Out ] = demoKoopmanForecastRKHS_nlsaModel( experiment ); 
toc( t )

nSX = getNTotalSample( model.embComponent );
idxT1 = getOrigin( model.embComponent );
idxPhi = getBasisFunctionIndices( model.koopmanOp );
nPhi = numel( idxPhi );
tauRKHS = getRegularizationParameter( model.koopmanOp );

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

%% READ TRAINING AND TEST DATA FOR KAF
% y is an array of size [ 3 nSY ] containing in its columns the response 
% variable (3-dimensional L63 state vector) on the training states. nSY is
% the number of response samples available after delay embedding.
%
% z is an array of size [ nSX nZ ] whose columns contain the eigenfunctions of
% the compactified generator on the training states. nSX is the number of 
% covariate samples available after delay embedding, and nZ the number of 
% generator eigenfunctions available from the nlsaModel. nSX is less than or 
% equal to nSY.  
%
% yO is an array of size [ 3 nSO ] containing in its columns the response
% variable on the test states. nSO is the number of test samples available
% after delay embedding.
%
% zO is an array of size [ nSO nPhi ] containing the out-of-sample values
% of the generator eigenfunctions.
if ifReadForecastData
    tic
    disp( 'Retrieving forecast data...' ); t = tic;
    
    % Response variable
    y = getData( model.trgComponent );
    y = y( :, idxT1 : end );

    % Eigenfunctions, eigenfrequencies, and expansion coefficients  
    [ z, mu ] = getKoopmanEigenfunctions( model ); 
    omega = getKoopmanEigenfrequencies( model );
    c = getEigenfunctionCoefficients( model.koopmanOp );

    % Response variable (test data)
    yO = getData( model.outTrgComponent );
    yO = yO( :, idxT1 : end );

    % Eigenfunctions (test data)
    phiO = getOseDiffusionEigenfunctions( model );
    zO = phiO( :, idxPhi ) * c;
    toc( t )
end

%% PERFORM FORECASTING
if ifForecast

    % Lead times
    tau = ( 0 : nT - 1 ) * In.dt; 

    tic 
    disp( 'Performing  forecast...' ); t = tic;
    yT = generatorForecast( y( :, 1 : nSX ), z( :, 1 : nL ), mu, ...
                            omega( 1 : nL  ), zO( :, 1 : nL ), tau );
    yT = real( yT ); % iron out numerical wrinkles

    % If idxPhi( 1 ) > 1 we are excluding the constant eigenfunction, so we 
    % compensate by adding the mean of y.
    if idxPhi( 1 ) > 1
        yT = yT + mean( y, 2 ); 
    end
    toc( t )

    tic
    disp( 'Computing forecast error...' ); t = tic;
    [ ~, yRMSE, yPC ] = forecastError( yO, yT );
    yRMSE = yRMSE ./ std( y, 0, 2 );
    toc( t )
end

%% PLOT FORECAST TRAJECTORIES
if ifPlotForecast

    % Lead times
    t = ( 0 : nT - 1 ) * In.dt; 

    % Plot limits and ticks
    yLm = [ [ -25 -25 5 ]' [ 25 25 50 ]' ]; 
    yTk = { -20 : 10 : 20, -20 : 10 : 20, 10 : 10 : 50 }; 
   
    % Set up figure and axes 
    Fig.nTileX     = 3;
    Fig.nTileY     = numel( idxTPlt );
    Fig.units      = 'inches';
    Fig.figWidth   = 8; 
    Fig.deltaX     = .6;
    Fig.deltaX2    = .15;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .3;
    Fig.gapX       = .4;
    Fig.gapY       = .3;
    Fig.gapT       = 0.32; 
    Fig.aspectR    = 3 / 4;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Plot skill scores
    for iT = 1 : Fig.nTileY; 

        % Initialization time, true and forecast signal
        t0 = ( idxTPlt( iT ) - 1 ) * In.dt;
        yTrue = yO( :, idxTPlt( iT ) : idxTPlt( iT ) + nT - 1 );
        yPred = squeeze( yT( :, :, idxTPlt( iT ) ) );


        % Loop over the components of the response vector
        for iY = 1 : 3
            set( gcf, 'currentAxes', ax( iY, iT ) )
            plot( t, yTrue( iY, : ), 'k-' )
            plot( t, yPred( iY, : ), 'b-' )

            xlim( [ t( 1 ) t( end ) ] )
            ylim( yLm( iY, : ) )
            set( gca, 'xTick', t( 1 ) : 1 : t( end ), ...
                      'yTick', yTk{ iY } )  
            grid on

            if iY == 1
                ylabel( sprintf( 'Init. time t_0 = %1.1f', t0 ) )
            end
            if iT == 1
                title( sprintf( 'x_{%i}', iY ) )
            end
            if iT == Fig.nTileY
                xlabel( 'Lead time t' ) 
            else
                set( gca, 'xTickLabel', [] )
            end
            if iT == 1 && iY == 2
                legend( 'true', 'forecast', 'location', 'northEast' )
            end
        end
    end


    % Add figure title
    titleStr = sprintf( [ '%s, regularization parameter %1.2g, ' ...
                          '# diffusion basis = %i, ', ...
                          '# generator eigenfuncs = %i' ], ...
                          tStr, tauRKHS,  nPhi, nL  );
    title( axTitle, titleStr )
            

    % Print figure
    if ifPrintFig
        figFile = sprintf( 'figForecastTrajectory_tau%1.2g,nL%i.png', nL );
        figFile = fullfile( figDir, figFile );
        print( fig, figFile, '-dpng', '-r300' ) 
    end

end

%% PLOT ERROR SCORES
if ifPlotError

    % Lead times
    t = ( 0 : nT - 1 ) * In.dt; 

    % Set up figure and axes 
    Fig.nTileX     = 3;
    Fig.nTileY     = 2;
    Fig.units      = 'inches';
    Fig.figWidth   = 8; 
    Fig.deltaX     = .6;
    Fig.deltaX2    = .15;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .3;
    Fig.gapX       = .3;
    Fig.gapY       = .3;
    Fig.gapT       = 0.32; 
    Fig.aspectR    = 3 / 4;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Plot skill scores
    for iY = 1 : 3 

        % Normalized RMSE
        set( gcf, 'currentAxes', ax( iY, 1 ) )
        plot( t, yRMSE( iY, : ) )

        grid on
        ylim( [ 0 1.6 ] )
        xlim( [ t( 1 ) t( end ) ] )
        set( gca, 'xTick', t( 1 ) : 1 : t( end ), 'yTick', 0 : .4 : 1.6, ...
                  'xTickLabel', [] )  
        if iY == 1
            ylabel( 'Normalized RMSE' )
        else
            set( gca, 'yTickLabel', [] )
        end
        title( sprintf( 'x_{%i}', iY ) )
        
        % Pattern correlation
        set( gcf, 'currentAxes', ax( iY, 2 ) )
        plot( t, yPC( iY, : ) )

        grid on
        ylim( [ -0.2 1.1 ] )
        xlim( [ t( 1 ) t( end ) ] )
        set( gca, 'xTick', t( 1 ) : 1 : t( end ), 'yTick', -0.2 : .2 : 1.2 )  
        if iY == 1
            ylabel( 'Pattern correlation' )
        else
            set( gca, 'yTickLabel', [] )
        end
        xlabel( 'Lead time t' ) 
    end

    % Add figure title
    titleStr = sprintf( [ '%s, regularization parameter %1.2g, ' ...
                          '# diffusion basis = %i, ', ...
                          '# generator eigenfuncs = %i' ], ...
                          tStr, tauRKHS,  nPhi, nL  );
    title( axTitle, titleStr )

    % Print figure
    if ifPrintFig
        figFile = sprintf( 'figForecastError_tau%1.2g,nL%i.png', nL );
        figFile = fullfile( figDir, figFile );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end
        



