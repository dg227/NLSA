% RECONSTRUCT THE LIFECYCLE OF THE EL NINO SOUTHERN OSCILLATION (ENSO) 
% USING DATA-DRIVEN SPECTRAL ANALYSIS OF KOOPMAN/TRANSFER OPERATORS
%
% Modified 2021/09/14

%% SCRIPT EXECUTION OPTIONS
ifDataSrc           = false; % source data (Indo-Pacific SST)
ifDataTrg           = false; % target data (Nino 3.4 index)
ifNLSA              = false; % perform NLSA (kernel eigenfunctions)
ifKoopman           = false; % compute Koopman eigenfunctions
ifNinoLifecycle     = true;  % ENSO lifecycle from Nino 3.4 index 
ifKoopmanLifecycle  = true;  % ENSO lifecycle from generator eigenfunction
ifPlotSpectrum      = true;  % plot spectrum of the generator
ifPlotEnsoLifecycle = true;
ifPrintFig          = true;  % print figures to file

%% NLSA PARAMETERS
% ERSSTv4 reanalysis
NLSA.dataset           = 'ersstV4';
NLSA.period            = { '197001' '202002' };
NLSA.climatologyPeriod = { '198101' '201012' };
NLSA.sourceVar         = { 'IPSST' };
NLSA.targetVar         = { 'Nino3.4' };
NLSA.embWindow         = 48; 
NLSA.kernel            = 'cone';
NLSA.ifDen             = false;

experiment = demoKoopman_experiment( NLSA );

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes


%% FIGURE DIRECTORY
figDir = fullfile( pwd, 'figs', experiment );
if ~isdir( figDir )
    mkdir( figDir )
end

%% EXTRACT SOURCE DATA
if ifDataSrc
    for iVar = 1 : numel( NLSA.sourceVar )
        msgStr = sprintf( 'Reading source data %s...', ...
                          NLSA.sourceVar{ iVar } );
        disp( msgStr ) 
        t = tic;
        demoKoopman_data( NLSA.dataset, NLSA.period, NLSA.sourceVar{ iVar }, ...
                          NLSA.climatologyPeriod ) 
        toc( t )
    end
end

%% EXTRACT TARGET DATA
if ifDataTrg
    for iVar = 1 : numel( NLSA.targetVar )
        msgStr = sprintf( 'Reading target data %s...', ...
                           NLSA.targetVar{ iVar } );
        disp( msgStr ) 
        t = tic;
        demoKoopman_data( NLSA.dataset, NLSA.period, NLSA.targetVar{ iVar }, ...
                          NLSA.climatologyPeriod ) 
        toc( t )
    end
end


%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES
% In is a data structures containing the NLSA parameters. 
%
% nSE is the number of training samples available after Takens delay
% embedding.
%
% nE is equal to half of the delay embedding window length employed in the 
% training data. 
%
% nSB is the number of samples left out in the start of the time interval (for
% temporal finite differnences employed in the kernel).
%
% nShiftTakens is the temporal shift applied to align Nino indices with the
% center of the Takens embedding window eployed in the NLSA kernel. 

[ model, In ] = demoKoopman_nlsaModel( NLSA );

nSE = getNTotalEmbSample( model );    
nSB = getNXB( model.embComponent );
nShiftTakens = floor( getEmbeddingWindow( model.embComponent ) / 2 );

%% SPECIFY GLOBAL PARAMETERS
% The following variables are defined:
% nShiftNino:   Temporal shift to obtain 2D Nino index
% idxZEnso:     ENSO eigenfunction from generator      
% phaseZ:       Phase multpiplication factor (for consistency with Nino)
% Spec:         Parameters for generator spectral plots
% iC<VarName>:  Index of variable VarName in the target data of the nlsaModel 
% iRec<Name>:   Index of Koopman-reconstructed mode Name in nlsaModel 

nShiftNino = 11;        
iCNino34   = 1;  % Nino 3.4 index

% Plot color for eigenvalue groups
Spec.Color.const   = [ 0 0 0 ];
Spec.Color.annual  = [ 0 0 1 ];
Spec.Color.semi    = [0.9290 0.6940 0.1250];
Spec.Color.tri     = [ 1 0 1 ];
Spec.Color.trend   = [ 0 1 0 ];
Spec.Color.trendC  = [ 0 1 1 ];
Spec.Color.enso    = [ 1 0 0 ];
Spec.Color.ensoC   = [0.4940 0.1840 0.5560]; 
Spec.Color.decadal = [ 1 1 0 ] * .5;

Spec.legend = { 'seasonal branch' ...
                'trend branch' ...
                'ENSO branch' ...
                'constant' ...
                'annual' ...
                'semiannual' ...
                'triannual' ...
                'trend' ... 
                'trend combination' ...
                'fundamental ENSO' ...
                'ENSO combination' ...
                'decadal' ...
               };


switch experiment

case 'ersstV4_197001-202002_IPSST_emb48_cone'

    datasetStr = 'ERSSTv4'; 

    % ENSO eigenfunction and phase multiplication factor
    idxZEnso     = 7;
    phaseZ       = exp( i * 5 * pi / 32 );        


    % Indices of eigenvalue groups in the spectrum
    Spec.Idx.const        = 1;
    Spec.Idx.annual       = [ 2 3 ];
    Spec.Idx.semi         = [ 4 5 ];
    Spec.Idx.tri          = [ 13 14 ];
    Spec.Idx.trend        = 6;
    Spec.Idx.trendC       = [ 9 10 ];
    Spec.Idx.enso         = [ 7 8 ];
    Spec.Idx.ensoC        = [ 11 12 16 17];
    Spec.Idx.decadal      = 15;
    Spec.Idx.annualBranch = [ 13 4 2 3 5 14 ];
    Spec.Idx.trendBranch  = [ 9 6 10 ];
    Spec.Idx.ensoBranch   = [ 16 11 7 8 12 17 ];

    Spec.xLim  = [ -1 .1 ];
    Spec.yLim  = [ -3 3 ]; 

otherwise
    error( 'Invalid experiment' )

end
    
%% PERFORM NLSA 
% Output from each step is saved on disk.
if ifNLSA

    disp( 'Takens delay embedding for source data...' ); t = tic; 
    computeDelayEmbedding( model )
    toc( t )

    % The following step is needed only if we are using velocity-dependent
    % kernels.
    if isa( model.embComponent, 'nlsaEmbeddedComponent_xi' )
        disp( 'Phase space velocity (time tendency of data)...' ); t = tic; 
        computeVelocity( model )
        toc( t )
    end

    % The following steps are needed only if we are using variable-bandwidth
    % kernels.
    if isa( model, 'nlsaModel_den' )
        fprintf( 'Pairwise distances for density data, %i/%i...\n', ...
                  iProc, nProc ); 
        t = tic;
        computeDenPairwiseDistances( model, iProc, nProc )
        toc( t )

        disp( 'Distance normalization for kernel density steimation...' );
        t = tic;
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
    end

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
    computeKoopmanEigenfunctions( model, 'ifLeftEigenfunctions', true )
    toc( t )
end

%% CONSTRUCT NINO-BASED ENSO LIFECYCLE
% Build a structure Nino34 such that:
% 
% Nino34.idx is an array of size [ 2 nSE ], where nSE is the number of samples 
% after delay embedding. Nino34.idx( 1, : ) contains the values of the 
% Nino 3.4 index at the current time. Nino34( 2, : ) contains the values of 
% the Nino 3.4 index at nShiftNino timesteps (months) in the past.
% 
% Nino34.time is an array of size [ 1 nSE ] containing the timestamps in
% Matlab serial date number format. 
if ifNinoLifecycle

    disp( 'Constructing Nino 3.4-based bivariate ENSO index...' ); t = tic;

    % Timestamps
    Nino34.time = getTrgTime( model ); 
    Nino34.time = Nino34.time( nSB + 1 + nShiftTakens : end );
    Nino34.time = Nino34.time( 1 : nSE );

    % Nino 3.4 index
    nino = getData( model.trgComponent( iCNino34 ) );
    Nino34.idx = [ nino( nShiftNino + 1 : end ) 
                 nino( 1 : end - nShiftNino ) ];
    Nino34.idx = Nino34.idx( :, nSB + nShiftTakens - nShiftNino + 1 : end );
    Nino34.idx = Nino34.idx( :, 1 : nSE );
    Nino34.angle = angle( complex( Nino34.idx( 1, : ), Nino34.idx( 2, : ) ) );
end

%% ENSO LIFECYCLE BASED ON KOOPMAN EIGENFUNCTIONS
% Build a structure Z such that:
% 
% Z.idx is a complex array of size [ 2 nSE ], where nSE is the number of 
% samples after delay embedding. Z.idx( 1, : ) and Z.idx( 2, : ) contain the 
% real and imaginary parts of the generator-based ENSO index. 
% 
% Z.time is an array of size [ 1 nSE ] containing the timestamps in
% Matlab serial date number format. 

if ifKoopmanLifecycle

    disp( 'Constructing Koopman-based complex ENSO index...' ); t = tic;

    % Retrieve Koopman eigenfunctions
    z = getKoopmanEigenfunctions( model );
    T = getEigenperiods( model.koopmanOp );
    TEnso = abs( T( idxZEnso ) / 12 );
    Z.idx = [ real( phaseZ * z( :, idxZEnso ) )' 
             imag( phaseZ * z( :, idxZEnso ) )' ];
    Z.time = getTrgTime( model );
    Z.time = Z.time( nSB + 1 + nShiftTakens : end );
    Z.time = Z.time( 1 : nSE );
    Z.angle = angle( complex( Z.idx( 1, : ), Z.idx( 2, :) ) );
end

%% PLOT GENERATOR SPECTRUM
if ifPlotSpectrum

    % Retrieve Koopman eigenvalues. Set frequency units to cycles/year.
    gamma = getKoopmanEigenvalues( model ) * 12 / 2 / pi; 

    % Determine which eigenvalues will get special marking
    ifMark = false( size( gamma ) );
    ifMark( [ Spec.Idx.const Spec.Idx.annual Spec.Idx.semi ...
              Spec.Idx.tri Spec.Idx.enso Spec.Idx.ensoC ...
              Spec.Idx.decadal ...
              Spec.Idx.trend Spec.Idx.trendC ] ) = true;

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 5; 
    Fig.deltaX     = .55;
    Fig.deltaX2    = 2.1;
    Fig.deltaY     = .55;
    Fig.deltaY2    = .2;
    Fig.gapX       = .25;
    Fig.gapY       = .5;
    Fig.gapT       = 0; 
    Fig.nTileX     = 1;
    Fig.nTileY     = 1;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );

    set( gcf, 'currentAxes', ax( 1 ) )

    % Plot marked eigenvalue groups
    plot( real( gamma( Spec.Idx.annualBranch ) ), ...
          imag( gamma( Spec.Idx.annualBranch ) ), 'b-' )
    plot( real( gamma( Spec.Idx.trendBranch ) ), ...
          imag( gamma( Spec.Idx.trendBranch ) ), 'g-' )
    plot( real( gamma( Spec.Idx.ensoBranch ) ), ...
          imag( gamma( Spec.Idx.ensoBranch ) ), 'r-' )
    plot( real( gamma( Spec.Idx.const ) ), ...
          imag( gamma( Spec.Idx.const ) ), ...
          '.', 'markersize', 15, 'color', Spec.Color.const )
    plot( real( gamma( Spec.Idx.annual ) ), ...
          imag( gamma( Spec.Idx.annual ) ), ...
          '.', 'markersize', 15, 'color', Spec.Color.annual )
    plot( real( gamma( Spec.Idx.semi ) ), ...
          imag( gamma( Spec.Idx.semi ) ), ...
          '.', 'markersize', 15, 'color', Spec.Color.semi )
    plot( real( gamma( Spec.Idx.tri ) ), ...
          imag( gamma( Spec.Idx.tri ) ), ...
          '.', 'markersize', 15, 'color', Spec.Color.tri )
    plot( real( gamma( Spec.Idx.trend ) ), ...
          imag( gamma( Spec.Idx.trend ) ), ...
          '<', 'markersize', 6, 'color', Spec.Color.trend, ...
          'markerFaceColor', Spec.Color.trend )
    plot( real( gamma( Spec.Idx.trendC ) ), ...
          imag( gamma( Spec.Idx.trendC ) ), ...
          's', 'markersize', 6, 'color', Spec.Color.trendC, ...
          'markerFaceColor', Spec.Color.trendC )
    plot( real( gamma( Spec.Idx.enso ) ), ...
          imag( gamma( Spec.Idx.enso ) ), ...
          '^', 'markersize', 5.5, 'color', Spec.Color.enso, ...
          'markerFaceColor', Spec.Color.enso )
    plot( real( gamma( Spec.Idx.ensoC ) ), ...
          imag( gamma( Spec.Idx.ensoC ) ), ...
          's', 'markersize', 6, 'color', Spec.Color.ensoC, ...
          'markerFaceColor', Spec.Color.ensoC )
    plot( real( gamma( Spec.Idx.decadal ) ), ...
          imag( gamma( Spec.Idx.decadal ) ), ...
          '>', 'markersize', 6, 'color', Spec.Color.decadal, ...
          'markerFaceColor', Spec.Color.decadal )

    % Plot unmarked eigenvalues
    plot( real( gamma( ~ifMark ) ), ...
          imag( gamma( ~ifMark ) ), '.', 'markerSize', 10, ...
          'color', [ .5 .5 .5 ] )

    % Add legend
    axPos = get( gca, 'position' );
    hL = legend( Spec.legend{ : }, 'location', 'eastOutside' );
    set( gca, 'position', axPos )
    lPos = get( hL, 'position' );
    lPos( 1 ) = lPos( 1 ) + .007; 
    set( hL, 'position', lPos );
    xlim( Spec.xLim );
    ylim( Spec.yLim ); 
    ylabel( 'Frequency \nu_j (cycles/yr)' )
    xlabel( 'Growth rate Re(\lambda_j)' ) 
    grid on
    title( sprintf( 'Generator spectrum -- %s', datasetStr ) )

    % Print figure
    if ifPrintFig
        figFile = fullfile( figDir, 'figGeneratorSpectrum.png' );
        print( fig, figFile, '-dpng', '-r600' ) 
    end
end


%% PLOT ENSO LIFECYCLE
if ifPlotEnsoLifecycle

    % Set time limits and tick intervals 
    idxTLim   = [ 1 nSE ];                         
    idxTLimTS = [ nSE - 360, nSE ];
    idxTick = idxTLimTS( 1 ) : 60 : idxTLimTS( 2 );


    % Significant El Nino events to show in time series plots
    idxNinos = [ find( Nino34.time == datenum( '011998', 'mmyyyy' ) ) ...
                 find( Nino34.time == datenum( '012016', 'mmyyyy' ) ) ...
                 find( Nino34.time == datenum( '111973', 'mmyyyy' ) ) ...
                 find( Nino34.time == datenum( '011999', 'mmyyyy' ) ) ];

    % Significant El Nino/La Nina events in phase space plots 
    ElNinos = { { '201511' '201603' } ... 
                { '199711' '199803' } ...
                { '199111' '199203' } ...
                { '198711' '198803' } ...
                { '198211' '198303' } ...
                { '197211' '197303' } ...
                { '196511' '196603' } ...
                { '195711' '195803' } };

    LaNinas = { { '201011' '201103' } ... 
                { '200711' '200803' } ...
                { '199911' '200003' } ...
                { '199811' '199903' } ...
                { '198811' '198903' } ...
                { '197511' '197603' } ...
                { '197309' '197401' } };


    % Amplitudes
    Nino34.ampl = abs( complex( Nino34.idx( 1, : ), Nino34.idx( 2, : ) ) );
    Z.ampl = abs( complex( Z.idx( 1, : ), Z.idx( 2, :) ) );

    % Set up figure and axes for lifecycle plots 
    Fig.units      = 'inches';
    Fig.figWidth   = 12; 
    Fig.deltaX     = .55;
    Fig.deltaX2    = .8;
    Fig.deltaY     = .6;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = .7;
    Fig.gapT       = .3; 
    Fig.nTileX     = 4;
    Fig.nTileY     = 2;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    txtPos = { 0.035 0.935 };
    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Plot Nino 3.4 lifecycle
    set( gcf, 'currentAxes', ax( 1, 1 ) )
    plotLifecycle( Nino34, ElNinos, LaNinas, model.tFormat, idxTLim )
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'xTick', -3 : 1 : 3, 'yTick', -3 : 1 : 3 )
    xlabel( 'Nino 3.4' )
    title( 'Phase space evolution' )
    text( txtPos{ : }, '(a)', 'units', 'normalized' )

    % Make scatterplot of Nino 3.4 lifecycle colored by Nino 3.4 index
    set( gcf, 'currentAxes', ax( 2, 1 ) )
    plot( Nino34.idx( 1, : ), Nino34.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Nino34.idx( 1, : ), Nino34.idx( 2, : ), 17, Nino34.idx( 1, : ), ...
             'o', 'filled' )  
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'clim', [ -3.2 3.2 ], 'xTick', [ -4 : 4 ], 'yTick', [ -4 : 4 ] )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    for iNino = 1 : numel( idxNinos )
        plot( Nino34.idx( 1, idxNinos( iNino ) ), ...
              Nino34.idx( 2, idxNinos( iNino ) ), ...
              'gp', 'markerSize', 10, 'linewidth', 2 )
        
        % This adjustment is made to improve readability
        if iNino == 3
           shift = -.7;
        else
          shift = .3;
        end 

        text( Nino34.idx( 1, idxNinos( iNino ) ) + shift, ...
              Nino34.idx( 2, idxNinos( iNino ) ), ...
              datestr( Nino34.time( idxNinos( iNino ) ), 'yy' ), ...
              'fontSize', 8, 'color', 'g' )
    end
    xlabel( 'Nino 3.4' )
    title( 'Evolution colored by Nino 3.4 index' )
    text( txtPos{ : }, '(b)', 'units', 'normalized', 'color', 'white' )

    % Plot Nino 3.4 time series
    set( gcf, 'currentAxes', ax( 3, 1 ) )
    plot( Nino34.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
          Nino34.idx( 1, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), 'b-' )
    ylim( [ -3 3 ] )
    xlim( [ Nino34.time( idxTLimTS( 1 ) ) Nino34.time( idxTLimTS( 2 ) ) ] ) 
    set( gca, 'xTick', Nino34.time( idxTick ), ...
              'xTickLabel', datestr( Nino34.time( idxTick ), 'yy' ) ) 
    grid on
    title( 'Time series' )
    legend( 'Nino 3.4', 'location', 'southWest' )
    text( txtPos{ : }, '(c)', 'units', 'normalized' )
    

    % Plot Nino 3.4 angle
    set( gcf, 'currentAxes', ax( 4, 1 ) )
    scatter( Nino34.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
             Nino34.angle( idxTLimTS( 1 ) : idxTLimTS( 2 ) ) / pi, ...
             17, Nino34.idx( 1, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
             'o', 'filled' )
    for iNino = 1 : numel( idxNinos )
        plot( Nino34.time( idxNinos( iNino ) ), ...
              Nino34.angle( idxNinos( iNino ) ) / pi, ...
              'gp', 'markerSize', 10, 'linewidth', 2 )
    end
    set( gca, 'color', [ 1 1 1 ] * .3 )
    ylim( [ -1.1 1.1 ] )
    xlim( [ Nino34.time( idxTLimTS( 1 ) ) Nino34.time( idxTLimTS( 2 ) ) ] ) 
    caxis( [ -3.2 3.2 ] )
    set( gca, 'xTick', Nino34.time( idxTick ), ...
              'xTickLabel', datestr( Nino34.time( idxTick ), 'yy' ), ...
              'yTick', [ - 1 : .5 : 1 ], 'yTickLabel', ...
              { '-\pi' '-\pi/2' '0', '\pi/2', '\pi' } ) 
    grid on
    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 3 ) = cPos( 3 ) * .7;
    cPos( 1 ) = cPos( 1 ) + .06;
    set( hC, 'position', cPos )
    xlabel( hC, 'Nino 3.4' )
    set( gca, 'position', axPos )
    title( 'Phase angle colored by Nino 3.4 index' )
    text( txtPos{ : }, '(d)', 'units', 'normalized', 'color', 'white' )


    % Plot generator lifecycle
    set( gcf, 'currentAxes', ax( 1, 2 ) )
    plotLifecycle( Z, ElNinos, LaNinas, model.tFormat, idxTLim )
    xlabel( sprintf( 'Re(g_{%i})', idxZEnso - 1 ) )
    ylabel( sprintf( 'Im(g_{%i}); eigenperiod = %1.2f y', idxZEnso - 1, TEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    text( txtPos{ : }, '(e)', 'units', 'normalized' )

    % Make scatterplot of generator lifecycle colored by Nino 3.4 index
    set( gcf, 'currentAxes', ax( 2, 2 ) )
    plot( Z.idx( 1, : ), Z.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Z.idx( 1, : ), Z.idx( 2, : ), 17, Nino34.idx( 1, : ), ...
             'o', 'filled' )  
    for iNino = 1 : numel( idxNinos )
        plot( Z.idx( 1, idxNinos( iNino ) ), ...
              Z.idx( 2, idxNinos( iNino ) ), 'gp', 'markerSize', 10, 'linewidth', 2 )

        % This adjustment is made to improve readability
        if iNino == 3
           shift = -.7;
        else
           shift = .3;
        end 
        text( Z.idx( 1, idxNinos( iNino ) ) + shift, ...
              Z.idx( 2, idxNinos( iNino ) ), ...
              datestr( Z.time( idxNinos( iNino ) ), 'yy' ), ...
              'fontSize', 8, 'color', 'g' )
    end
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    caxis( [ -3.2 3.2 ] )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    set( gca, 'xTick', [ -2 : 1 : 2 ], 'yTick', [ -2 : 1 : 2 ] )

    xlabel( sprintf( 'Re(g_{%i})', idxZEnso - 1 ) )
    text( txtPos{ : }, '(f)', 'units', 'normalized', 'color', 'white' )

    % Plot generator time series
    set( gcf, 'currentAxes', ax( 3, 2 ) )
    plot( Z.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
          Z.idx( 1, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), 'b-' )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'xTick', Z.time( idxTick ), ...
              'xTickLabel', datestr( Z.time( idxTick ), 'yy' ) ) 
    grid on
    legend( sprintf( 'Re(g_{%i})', idxZEnso - 1 ), ...
            'location', 'southWest' )
    xlim( [ Z.time( idxTLimTS( 1 ) ) Z.time( idxTLimTS( 2 ) ) ] ) 
    text( txtPos{ : }, '(g)', 'units', 'normalized' )

    % Plot generator angle
    set( gcf, 'currentAxes', ax( 4, 2 ) )
    scatter( Z.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
             Z.angle( idxTLimTS( 1 ) : idxTLimTS( 2 ) ) / pi, ...
             17, Nino34.idx( 1, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
             'o', 'filled' )
    for iNino = 1 : numel( idxNinos )
        plot( Z.time( idxNinos( iNino ) ), ...
              Z.angle( idxNinos( iNino ) ) / pi, ...
              'gp', 'markerSize', 10, 'linewidth', 2 )
    end
    ylim( [ -1.1 1.1 ] )
    xlim( [ Z.time( idxTLimTS( 1 ) ) Z.time( idxTLimTS( 2 ) ) ] ) 
    caxis( [ -3.2 3.2 ] )
    grid on
    set( gca, 'xTick', Nino34.time( idxTick ), ...
              'xTickLabel', datestr( Nino34.time( idxTick ), 'yy' ), ...
              'yTick', [ - 1 : .5 : 1 ], 'yTickLabel', ...
              { '-\pi' '-\pi/2' '0', '\pi/2', '\pi' } ) 
    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 3 ) = cPos( 3 ) * .7;
    cPos( 1 ) = cPos( 1 ) + .06;
    set( hC, 'position', cPos )
    xlabel( hC, 'Nino 3.4' )
    set( gca, 'position', axPos )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    text( txtPos{ : }, '(h)', 'units', 'normalized', 'color', 'white' )
    title( axTitle, sprintf( 'ENSO lifecycle -- %s', datasetStr ) ) 

    set( gcf, 'invertHardCopy', 'off' )
    figFile = fullfile( figDir, 'figEnsoLifecycle.png' );
    print( fig, figFile, '-dpng', '-r600' ) 

end


%% AUXILIARY FUNCTION TO PLOT LIFECYCLE
function plotLifecycle( Index, Ninos, Ninas, tFormat, idxTLim )

if nargin < 5
    idxTLim = [ 1 size( Index.idx, 2 ) ];
end 


% Plot temporal evolution of index
plot( Index.idx( 1, idxTLim( 1 ) : idxTLim( 2 ) ), ...
      Index.idx( 2, idxTLim( 1 ) : idxTLim( 2 ) ), 'g-' )
hold on
grid on

% Highlight significant El Ninos
for iENSO = 1 : numel( Ninos )

    % Serial date numbers for start and end of event
    tLim = datenum( Ninos{ iENSO }( 1 : 2 ), tFormat );
    
    % Find and plot portion of index time series
    idxT1     = find( Index.time == tLim( 1 ) );
    idxT2     = find( Index.time == tLim( 2 ) );
    idxTLabel = round( ( idxT1 + idxT2 ) / 2 ); 
    plot( Index.idx( 1, idxT1 : idxT2 ), Index.idx( 2, idxT1 : idxT2 ), ...
          'r-', 'lineWidth', 2 )
    text( Index.idx( 1, idxTLabel ), Index.idx( 2, idxTLabel ), ...
          datestr( Index.time( idxT2 ), 'yy' ), 'fontSize', 8 )
end

% Highlight significant La Ninas
for iENSO = 1 : numel( Ninas )

    % Serial date numbers for start and end of event
    tLim = datenum( Ninas{ iENSO }( 1 : 2 ), tFormat );
    
    % Find and plot portion of index time series
    idxT1 = find( Index.time == tLim( 1 ) );
    idxT2 = find( Index.time == tLim( 2 ) );
    idxTLabel = round( ( idxT1 + idxT2 ) / 2 ); 
    plot( Index.idx( 1, idxT1 : idxT2 ), Index.idx( 2, idxT1 : idxT2 ), ...
          'b-', 'lineWidth', 2 )
    text( Index.idx( 1, idxTLabel ), Index.idx( 2, idxTLabel ), ...
          datestr( Index.time( idxT2 ), 'yy' ), 'fontSize', 8 )
end

end

