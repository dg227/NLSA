% RECONSTRUCT THE LIFECYCLE OF THE EL NINO SOUTHERN OSCILLATION (ENSO) 
% USING DATA-DRIVEN SPECTRAL ANALYSIS OF KOOPMAN/TRANSFER OPERATORS
%
% Modified 2020/05/13

%% DATA SPECIFICATION 
dataset   = 'ccsm4Ctrl';    % CCSM4 pre-industrial control run
period    = '200yr';        % 200-year analysis 
%period  = '1300yr';        % 1300-year analysis
sourceVar = 'IPSST_4yrEmb'; % Indo-Pacific SST, 4-year delay embedding 

%dataset    = 'noaa';                      % NOAA 20th Century Reanalysis 
%period    = 'satellite';                  % 1970-present
%experiment = 'enso_lifecycle_satellite';  % 1870-present 
%sourceVar = 'IPSST_4yrEmb'; % Indo-Pacific SST, 4-year delay embedding 

%% SCRIPT EXECUTION OPTIONS

% Data extraction
ifDataSST    = false;  % extract SST data from NetCDF source files
ifDataSSH    = false;  % extract SSH data from NetCDF source files
ifDataSAT    = false;  % extract SAT data from NetCDF source files
ifDataPrecip = false;  % extract precipitation data from NetCDF source files  
ifDataWind   = false;   % extract 10m wind data from NetCDF source files  

% ENSO representations
ifNLSA    = false; % compute kernel (NLSA) eigenfunctions
ifKoopman = false; % compute Koopman eigenfunctions
ifNinoIdx = false; % compute two-dimensional (lead/lag) Nino indices  

% ENSO 2D lifecycle plots
ifNLSALifecycle    = false;  % plot ENSO lifecycle from kernel eigenfunctions
ifKoopmanLifecycle = false; % plot ENSO lifecycle from generator eigenfuncs. 

% Lifecycle phases and equivariance plots
ifNLSAPhases          = false; % ENSO phases fron kerenel eigenfunctions
ifKoopmanPhases       = false; % ENSO phases from generator eigenfunctions
ifNLSAEquivariance    = false; % ENSO equivariance plots based on NLSA
ifKoopmanEquivariance = false; % ENSO equivariance plots based on Koopman

% Composite plots
ifNinoComposites    = true; % compute phase composites based on Nino 3.4 index
ifNLSAComposites    = true; % compute phase composites based on NLSA
ifKoopmanComposites = true; % compute phase composites based on Koopman

% Output/plotting options
ifPlotWind       = true;    % overlay quiver plot of surface winds 
ifPrintFig       = true;    % print figures to file
compositesDomain = 'globe'; % global domain
compositesDomain = 'Pacific'; % Pacific


%% GLOBAL PARAMETERS
% The following variables are defined:
% experiment:   String identifier for data analysis experiment
% nShiftNino:   Temporal shift to obtain 2D Nino index
% phase0:       Start phase in equivariance plots
% leads:        Leads (in months) for equivariance plots
% idxPhiEnso:   ENSO eigenfunctions from NLSA (kernel operator)
% signPhi:      Multiplication factor (for consistency with Nino)
% idxZEnso:     ENSO eigenfunction fro generator      
% phaseZ:       Phase multpiplication factor (for consistency with Nino)
% nSamplePhase: Number of samples per ENSO phase
% figDir:       Output directory for plots

nShiftNino   = 11;        
phase0       = 4;         
leads        = [ 0 6 12 18 24 ]; 

experiment = { dataset period sourceVar };
experiment = strjoin_e( experiment, '_' );


switch experiment
        
% NOAA 20th Century Reanalysis, industrial era, Indo-Pacific SST input,
% 4-year delay embeding window  
case 'noaa_industrial_IPSST_4yrEmb'

    idxPhiEnso   = [ 10 9 ];  
    signPhi      = [ -1 -1 ]; 
    idxZEnso     = 9;         
    phaseZ        = -1;        
    nPhase       = 8;         
    nSamplePhase = 100;       
    PRate.scl     = 1E5; 

% NOAA 20th Century Reanalysis, approximate satellite era, Indo-Pacific SST 
% input, 4-year delay embeding window  
case 'noaa_satellite_IPSST_4yrEmb'

    idxPhiEnso   = [ 8 7 ];  
    signPhi      = [ 1 -1 ]; 
    idxZEnso     = 7;         
    phaseZ       = -1 * exp( i * pi / 4 );        
    nPhase       = 8;         
    nSamplePhase = 30;       
    PRate.scl     = 1E5; 


% CCSM4 pre-industrial control, 200-year period, Indo-Pacific SST input, 
% 4-year delay embeding window  
case 'ccsm4Ctrl_200yr_IPSST_4yrEmb'

    idxPhiEnso   = [ 7 6 ];  
    signPhi      = [ 1 1 ]; 
    idxZEnso     = 6;         
    phaseZ       = i;        
    nPhase       = 8;         
    nSamplePhase = 100;       

% CCSM4 pre-industrial control, 1300-year period, Indo-Pacific SST input, 
% 4-year delay embeding window  
case 'ccsm4Ctrl_1300yr_IPSST_4yrEmb'

    idxPhiEnso   = [ 9 8 ];  
    signPhi      = [ -1 1 ]; 
    idxZEnso     = 8;         
    phaseZ       = exp( i * pi * ( 17 / 32 + 1 ) );        
    nPhase       = 8;         
    nSamplePhase = 200;       

otherwise
    error( 'Invalid experiment' )

end

% Figure directory
figDir = fullfile( pwd, 'figs', experiment );
if ~isdir( figDir )
    mkdir( figDir )
end



%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% EXTRACT SST DATA
if ifDataSST
 
    disp( 'Reading Indo-Pacific SST data...' ); t = tic;
    ensoLifecycle_data( dataset, period, 'IPSST' ) 
    toc( t )

    disp( 'Reading Nino 3.4 index...' ); t = tic;
    ensoLifecycle_data( dataset, period, 'Nino3.4' ) 
    toc( t )

    disp( 'Reading Nino 3 index...' ); t = tic;
    ensoLifecycle_data( dataset, period, 'Nino3' ) 
    toc( t )

    disp( 'Reading Nino 4 index...' ); t = tic;
    ensoLifecycle_data( dataset, period, 'Nino4' ) 
    toc( t )

    disp( 'Reading Nino 1+2 index...' ); t = tic;
    ensoLifecycle_data( dataset, period, 'Nino1+2' ) 
    toc( t )

    disp( 'Reading global SST data...' ); t = tic; 
    ensoLifecycle_data( dataset, period, 'SST' ) 
    toc( t )
end

%% EXTRACT SAT DATA
if ifDataSSH

    disp( 'Reading global SSH data...' ); t = tic; 
    ensoLifecycle_data( dataset, period, 'SSH' )
    toc( t )
end


%% EXTRACT SAT DATA
if ifDataSAT

    disp( 'Reading global SAT data...' ); t = tic; 
    ensoLifecycle_data( dataset, period, 'SAT' )
    toc( t )
end

%% EXTRACT PRECIPITATION RATE DATA
if ifDataPrecip

    disp( 'Reading global precipitation rate data...' ); t = tic; 
    ensoLifecycle_data( dataset, period, 'precip' )
    toc( t )
end

%% EXTRACT SURFACE WIND DATA
if ifDataWind
    disp( 'Reading global zonal surface wind data...' ); t = tic; 
    ensoLifecycle_data( dataset, period, 'uwind' )
    toc( t )

    disp( 'Reading global meridional surface wind data...' ); t = tic; 
    ensoLifecycle_data( dataset, period, 'vwind' )
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
% nShiftTakens is the temporal shift applied to align Nino indices with the
% center of the Takens embedding window eployed in the NLSA kernel. 

disp( 'Building NLSA model...' ); t = tic;
[ model, In ] = ensoLifecycle_nlsaModel( experiment ); 
toc( t )

nSE          = getNTotalSample( model.embComponent );
nSB          = getNXB( model.embComponent );
nShiftTakens = floor( getEmbeddingWindow( model.embComponent ) / 2 );

% Specify the NLSA model components corresponding to the analyzed observables
iCNino34 = 1;  % Nino 3.4 index
iCNino4  = 2;  % Nino 4 index
iCNino3  = 3;  % Nino 3 index
iCNino12 = 4;  % Nino 1+2 index
iCSST    = 5;  % global SST
iCSSH    = 6;  % global SSH
iCSAT    = 7;  % global SAT
iCPRate  = 8;  % global precipitation rate
iCUWnd   = 9;  % global surface meridional winds
iCVWnd   = 10; % global surface zonal winds

%% PERFORM NLSA
if ifNLSA
    
    % Execute NLSA steps. Output from each step is saved on disk

    disp( 'Takens delay embedding...' ); t = tic; 
    computeDelayEmbedding( model )
    toc( t )

    disp( 'Phase space velocity (time tendency of data)...' ); t = tic; 
    computeVelocity( model )
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

%% COMPUTE EIGENFUNCTIONS OF KOOPMAN GENERATOR
if ifKoopman
    disp( 'Koopman eigenfunctions...' ); t = tic;
    computeKoopmanEigenfunctions( model )
    toc( t )
end

%% CONSTRUCT TWO-DIMENSIONAL NINO INDEX
% Build a data structure Nino34 such that:
% 
% Nino34.idx is an array of size [ 2 nSE ], where nSE is the number of samples 
% after delay embedding. Nino34.idx( 1, : ) contains the values of the 
% Nino 3.4 index at the current time. Nino34( 2, : ) contains the values of 
% the Nino 3.4 index at nShiftNino timesteps (months) in the past.
% 
% Nino34.time is an array of size [ 1 nSE ] containing the timestamps in
% Matlab serial date number format. 
%
% Data stuctures Nino4, Nino3, Nino12 are constructed analogously for the 
% Nino 4, Nino 3, and Nino 1+2 indices, respectively. 
if ifNinoIdx

    disp( 'Constructing lagged Nino indicesx...' ); t = tic;

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


    % Nino 4 index
    Nino4.time = Nino34.time;
    nino = getData( model.trgComponent( iCNino4 ) );
    Nino4.idx = [ nino( nShiftNino + 1 : end ) 
                 nino( 1 : end - nShiftNino ) ];
    Nino4.idx = Nino4.idx( :, nSB + nShiftTakens - nShiftNino + 1 : end );
    Nino4.idx = Nino4.idx( :, 1 : nSE );

    % Nino 3 index
    Nino3.time = Nino34.time;
    nino = getData( model.trgComponent( iCNino3 ) );
    Nino3.idx = [ nino( nShiftNino + 1 : end ) 
                 nino( 1 : end - nShiftNino ) ];
    Nino3.idx = Nino3.idx( :, nSB + nShiftTakens - nShiftNino + 1 : end );
    Nino3.idx = Nino3.idx( :, 1 : nSE );

    % Nino 1+2 index
    Nino12.time = Nino34.time;
    nino = getData( model.trgComponent( iCNino12 ) );
    Nino12.idx = [ nino( nShiftNino + 1 : end ) 
                 nino( 1 : end - nShiftNino ) ];
    Nino12.idx = Nino12.idx( :, nSB + nShiftTakens - nShiftNino + 1 : end );
    Nino12.idx = Nino12.idx( :, 1 : nSE );

    toc( t );
end

%% PARAMETERS FOR ENSO LIFEYCLE PLOTS

% Plot limits for Nino indices
switch dataset
    
case 'noaa'

    PlotLim.nino4  = [ -3 3 ];
    PlotLim.nino34 = [ -3 3 ];
    PlotLim.nino3  = [ -3 3 ];
    PlotLim.nino12 = [ -4 4 ];

case 'ccsm4Ctrl'

    PlotLim.nino4  = [ -3.5 3.5 ];
    PlotLim.nino34 = [ -4 4 ];
    PlotLim.nino3  = [ -4 4 ];
    PlotLim.nino12 = [ -4.5 4.5 ];

otherwise
    error( 'Invalid dataset' )
end

% El Nino/La Nina events to mark up in lifecycle plots (in yyyymm format)
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
            { '197311' '197403' } };


%% PLOT ENSO LIFECYCLE BASED ON NLSA EIGENFUNCTIONS
if ifNLSALifecycle

    % Retrieve NLSA eigenfunctions
    phi = getDiffusionEigenfunctions( model );
    Phi.idx = ( signPhi .* phi( :, idxPhiEnso ) )';
    Phi.time = getTrgTime( model );
    Phi.time = Phi.time( nSB + 1 + nShiftTakens : end );
    Phi.time = Phi.time( 1 : nSE );
    
    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 15; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .65;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = .3;
    Fig.gapT       = 0; 
    Fig.nTileX     = 5;
    Fig.nTileY     = 2;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );

    % Plot Nino 4 lifecycle
    set( gcf, 'currentAxes', ax( 1, 1 ) )
    plotLifecycle( Nino4, ElNinos, LaNinas, model.tFormat )
    %xlabel( 'Nino 4' )
    ylabel( sprintf( 'Nino - %i months', nShiftNino ) )
    xlim( PlotLim.nino4 )
    ylim( PlotLim.nino4 )
    title( 'Nino 4 lifecycle' )

    % Plot Nino 3.4 lifecycle
    set( gcf, 'currentAxes', ax( 2, 1 ) )
    plotLifecycle( Nino34, ElNinos, LaNinas, model.tFormat )
    %xlabel( 'Nino 3.4' )
    %ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( PlotLim.nino34 )
    ylim( PlotLim.nino34 )
    title( 'Nino 3.4 lifecycle' )

    % Plot Nino 3 lifecycle
    set( gcf, 'currentAxes', ax( 3, 1 ) )
    plotLifecycle( Nino3, ElNinos, LaNinas, model.tFormat )
    %xlabel( 'Nino 3' )
    %ylabel( sprintf( 'Nino 3 - %i months', nShiftNino ) )
    xlim( PlotLim.nino3 )
    ylim( PlotLim.nino3 )
    title( 'Nino 3 lifecycle' )

    % Plot Nino 1+2 lifecycle
    set( gcf, 'currentAxes', ax( 4, 1 ) )
    plotLifecycle( Nino12, ElNinos, LaNinas, model.tFormat )
    %xlabel( 'Nino 3' )
    %ylabel( sprintf( 'Nino 1+2 - %i months', nShiftNino ) )
    xlim( PlotLim.nino12 )
    ylim( PlotLim.nino12 )
    title( 'Nino 1+2 lifecycle' )

    % Plot NLSA lifecycle
    set( gcf, 'currentAxes', ax( 5, 1 ) )
    plotLifecycle( Phi, ElNinos, LaNinas, model.tFormat )
    xlabel( sprintf( '\\phi_{%i}', idxPhiEnso( 1 ) ) )
    ylabel( sprintf( '\\phi_{%i}', idxPhiEnso( 2 ) ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'yAxisLocation', 'right' )
    title( 'NLSA lifecycle' )

    % Make scatterplot of NLSA lifcycle colored by Nino 4 index
    set( gcf, 'currentAxes', ax( 1, 2 ) )
    plot( Phi.idx( 1, : ), Phi.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Phi.idx( 1, : ), Phi.idx( 2, : ), 17, Nino4.idx( 1, : ), ...
             'o', 'filled' )  
    xlabel( sprintf( '\\phi_{%i}', idxPhiEnso( 1 ) ) )
    ylabel( sprintf( '\\phi_{%i}', idxPhiEnso( 2 ) ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'clim', [ -1 1 ] * 2.5 )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )

    % Make scatterplot of NLSA lifcycle colored by Nino 3.4 index
    set( gcf, 'currentAxes', ax( 2, 2 ) )
    plot( Phi.idx( 1, : ), Phi.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Phi.idx( 1, : ), Phi.idx( 2, : ), 17, Nino34.idx( 1, : ), ...
             'o', 'filled' )  
    xlabel( sprintf( '\\phi_{%i}', idxPhiEnso( 1 ) ) )
    %ylabel( sprintf( '\\phi_{%i}', idxPhiEnso( 2 ) ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'clim', [ -1 1 ] * 2.5 )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )

    % Make scatterplot of NLSA lifcycle colored by Nino 3 index
    set( gcf, 'currentAxes', ax( 3, 2 ) )
    plot( Phi.idx( 1, : ), Phi.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Phi.idx( 1, : ), Phi.idx( 2, : ), 17, Nino3.idx( 1, : ), ...
             'o', 'filled' )  
    xlabel( sprintf( '\\phi_{%i}', idxPhiEnso( 1 ) ) )
    %ylabel( sprintf( '\\phi_{%i}', idxPhiEnso( 2 ) ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'clim', [ -1 1 ] * 2.5 )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )

    % Make scatterplot of NLSA lifcycle colored by Nino 1+2 index
    set( gcf, 'currentAxes', ax( 4, 2 ) )
    plot( Phi.idx( 1, : ), Phi.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Phi.idx( 1, : ), Phi.idx( 2, : ), 17, Nino12.idx( 1, : ), ...
             'o', 'filled' )  
    xlabel( sprintf( '\\phi_{%i}', idxPhiEnso( 1 ) ) )
    %ylabel( sprintf( '\\phi_{%i}', idxPhiEnso( 2 ) ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'clim', [ -1 1 ] * 2.5 )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 3 ) = cPos( 3 ) * .7;
    cPos( 1 ) = cPos( 1 ) + .045;
    set( hC, 'position', cPos )
    xlabel( hC, 'Nino index' )
    set( gca, 'position', axPos )

    % Make redundant axis invisible
    set( gcf, 'currentAxes', ax( 5, 2 ) )
    axis off

    % Print figure
    if ifPrintFig
        set( gcf, 'invertHardCopy', 'off' )
        figFile = fullfile( figDir, 'figEnsoLifecycleKernel.png' );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end

%% PLOT ENSO LIFECYCLE BASED ON KOOPMAN EIGENFUNCTIONS
if ifKoopmanLifecycle

    % Retrieve Koopman eigenfunctions
    z = getKoopmanEigenfunctions( model );
    T = getEigenperiods( model.koopmanOp );
    TEnso = abs( T( idxZEnso ) / 12 );
    Z.idx = [ real( phaseZ * z( :, idxZEnso ) )' 
             imag( phaseZ * z( :, idxZEnso ) )' ];
    Z.time = getTrgTime( model );
    Z.time = Z.time( nSB + 1 + nShiftTakens : end );
    Z.time = Z.time( 1 : nSE );
    
    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 15; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .65;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = .3;
    Fig.gapT       = 0; 
    Fig.nTileX     = 5;
    Fig.nTileY     = 2;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );

    % Plot Nino 4 lifecycle
    set( gcf, 'currentAxes', ax( 1, 1 ) )
    plotLifecycle( Nino4, ElNinos, LaNinas, model.tFormat )
    %xlabel( 'Nino 4' )
    ylabel( sprintf( 'Nino - %i months', nShiftNino ) )
    xlim( PlotLim.nino4 )
    ylim( PlotLim.nino4 )
    title( 'Nino 4 lifecycle' )

    % Plot Nino 3.4 lifecycle
    set( gcf, 'currentAxes', ax( 2, 1 ) )
    plotLifecycle( Nino34, ElNinos, LaNinas, model.tFormat )
    %xlabel( 'Nino 3.4' )
    %ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( PlotLim.nino34 )
    ylim( PlotLim.nino34 )
    title( 'Nino 3.4 lifecycle' )

    % Plot Nino 3 lifecycle
    set( gcf, 'currentAxes', ax( 3, 1 ) )
    plotLifecycle( Nino3, ElNinos, LaNinas, model.tFormat )
    %xlabel( 'Nino 3' )
    %ylabel( sprintf( 'Nino 3 - %i months', nShiftNino ) )
    xlim( PlotLim.nino3 )
    ylim( PlotLim.nino3 )
    title( 'Nino 3 lifecycle' )

    % Plot Nino 1+2 lifecycle
    set( gcf, 'currentAxes', ax( 4, 1 ) )
    plotLifecycle( Nino12, ElNinos, LaNinas, model.tFormat )
    %xlabel( 'Nino 3' )
    %ylabel( sprintf( 'Nino 1+2 - %i months', nShiftNino ) )
    xlim( PlotLim.nino12 )
    ylim( PlotLim.nino12 )
    title( 'Nino 1+2 lifecycle' )


    % Plot generator lifecycle
    set( gcf, 'currentAxes', ax( 5, 1 ) )
    plotLifecycle( Z, ElNinos, LaNinas, model.tFormat )
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    ylabel( sprintf( 'Im(z_{%i})', idxZEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'yAxisLocation', 'right' )
    title( sprintf( 'Koopman lifecycle; eigenperiod = %1.2f y', TEnso ) )

    % Make scatterplot of generator lifcycle colored by Nino 4 index
    set( gcf, 'currentAxes', ax( 1, 2 ) )
    plot( Z.idx( 1, : ), Z.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Z.idx( 1, : ), Z.idx( 2, : ), 17, Nino4.idx( 1, : ), ...
             'o', 'filled' )  
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    ylabel( sprintf( 'Im(z_{%i})', idxZEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'clim', [ -1 1 ] * 2.5 )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )

    % Make scatterplot of generator lifcycle colored by Nino 3.4 index
    set( gcf, 'currentAxes', ax( 2, 2 ) )
    plot( Z.idx( 1, : ), Z.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Z.idx( 1, : ), Z.idx( 2, : ), 17, Nino34.idx( 1, : ), ...
             'o', 'filled' )  
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'clim', [ -1 1 ] * 2.5 )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )

    % Make scatterplot of generator lifcycle colored by Nino 3 index
    set( gcf, 'currentAxes', ax( 3, 2 ) )
    plot( Z.idx( 1, : ), Z.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Z.idx( 1, : ), Z.idx( 2, : ), 17, Nino3.idx( 1, : ), ...
             'o', 'filled' )  
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'clim', [ -1 1 ] * 2.5 )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )

    % Make scatterplot of generator lifcycle colored by Nino 1+2 index
    set( gcf, 'currentAxes', ax( 4, 2 ) )
    plot( Z.idx( 1, : ), Z.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Z.idx( 1, : ), Z.idx( 2, : ), 17, Nino12.idx( 1, : ), ...
             'o', 'filled' )  
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    set( gca, 'clim', [ -1 1 ] * 2.5 )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'eastOutside' );
    cPos = get( hC, 'position' );
    cPos( 3 ) = cPos( 3 ) * .7;
    cPos( 1 ) = cPos( 1 ) + .045;
    set( hC, 'position', cPos )
    xlabel( hC, 'Nino index' )
    set( gca, 'position', axPos )

    % Make redundant axis invisible
    set( gcf, 'currentAxes', ax( 5, 2 ) )
    axis off

    % Print figure
    if ifPrintFig
        figFile = fullfile( figDir, 'figEnsoLifecycleGenerator.png' );
        set( gcf, 'invertHardCopy', 'off' )
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end

%% COMPUTE AND PLOT ENSO PHASES BASED ON NLSA EIGENFUNCTIONS
%
% selectIndPhi is a cell array of size [ 1 nPhase ]. selectIndNLSA{ iPhase } 
% is a row vector containing the indices (timestamps) of the data affiliated
% with ENSO phase iPHase. 
%
% anglesPhi is a row vector of size [ 1 nPhase ] containing the polar angles
% in the 2D plane of the phase boundaries.
% 
% avNino34IndPhi is a row vector of size [ 1 nPhase ] containing the average
% Nino 3.4 index for each NLSA phase. 
%
% selectIndNino34, anglesNino34, and avNino34IndNino34 are defined analogously to
% selectIndPhi, anglesPhi, and avNino34IndPhi, respectively, using the Nino 3.4
% index. 
if ifNLSAPhases
   
    % Compute ENSO phases based on NLSA
    [ selectIndPhi, anglesPhi, avNino34IndPhi ] = computeLifecyclePhases( ...
        Phi.idx', Nino34.idx( 1, : )', nPhase, nSamplePhase );

    % Compute ENSO phases based on Nino 3.4 index
    [ selectIndNino34, anglesNino34, avNino34IndNino34 ] = ...
        computeLifecyclePhases( Nino34.idx', Nino34.idx(1,:)', nPhase, ... 
                                nSamplePhase );
        
    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 8; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .1;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .3;
    Fig.gapX       = .60;
    Fig.gapY       = .3;
    Fig.gapT       = 0; 
    Fig.nTileX     = 2;
    Fig.nTileY     = 1;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 8;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );

    % Plot Nino 3.4 phases
    set( gcf, 'currentAxes', ax( 1 ) )
    plotPhases( Nino34.idx', selectIndNino34, anglesNino34 ) 
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( PlotLim.nino34 )
    ylim( PlotLim.nino34 )

    % Plot NLSA phases
    set( gcf, 'currentAxes', ax( 2 ) )
    plotPhases( Phi.idx', selectIndPhi, anglesPhi )
    xlabel( sprintf( '\\phi_{%i}', idxPhiEnso( 1 ) ) )
    ylabel( sprintf( '\\phi_{%i}', idxPhiEnso( 2 ) ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    title( 'Kernel integral operator' )

    % Print figure
    if ifPrintFig
        figFile = fullfile( figDir, 'figEnsoPhasesKernel.png' );
        print( fig, figFile, '-dpng', '-r300' ) 
    end

end

%% COMPUTE AND PLOT ENSO PHASES BASED ON GENERATOR EIGENFUNCTIONS
%
% selectIndZ is a cell array of size [ 1 nPhase ]. selectIndZ{ iPhase } 
% is a row vector containing the indices (timestamps) of the data affiliated
% with ENSO phase iPHase. 
%
% anglesZ is a row vector of size [ 1 nPhase ] containing the polar angles
% in the 2D plane of the phase boundaries.
% 
% avNino34IndZ is a row vector of size [ 1 nPhase ] containing the average
% Nino 3.4 index for each NLSA generator. 
%
% selectIndNino34, anglesNino34, and avNino34IndNino34 are defined analogously to
% selectIndZ, anglesZ, and avNino34IndZ, respectively, using the Nino 3.4
% index. 
if ifKoopmanPhases
   
    % Compute ENSO phases based on generator
    [ selectIndZ, anglesZ, avNino34IndZ ] = computeLifecyclePhases( ...
        Z.idx', Nino34.idx( 1, : )', nPhase, nSamplePhase );

    % Compute ENSO phases based on Nino 3.4 index
    [ selectIndNino34, anglesNino34, avNino34IndNino34 ] = ...
        computeLifecyclePhases( Nino34.idx', Nino34.idx( 1, : )', ...
        nPhase, nSamplePhase );
        
    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 8; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .1;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .3;
    Fig.gapX       = .60;
    Fig.gapY       = .3;
    Fig.gapT       = 0; 
    Fig.nTileX     = 2;
    Fig.nTileY     = 1;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 8;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );

    % Plot Nino 3.4 phases
    set( gcf, 'currentAxes', ax( 1 ) )
    plotPhases( Nino34.idx', selectIndNino34, anglesNino34 ) 
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( PlotLim.nino34 )
    ylim( PlotLim.nino34 )

    % Plot generator phases
    set( gcf, 'currentAxes', ax( 2 ) )
    plotPhases( Z.idx', selectIndZ, anglesZ )
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    ylabel( sprintf( 'Im(z_{%i})', idxZEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    title( sprintf( 'Generator; eigenperiod = %1.2f y', TEnso ) )

    % Print figure
    if ifPrintFig
        figFile = fullfile( figDir, 'figEnsoPhasesKoopman.png' );
        print( fig, figFile, '-dpng', '-r300' ) 
    end


end

%% EQUIVARIANCE PLOTS BASED ON NLSA
if ifNLSAEquivariance

    nLead = numel( leads );  

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 10; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .1;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .5;
    Fig.gapX       = .20;
    Fig.gapY       = .5;
    Fig.gapT       = .25; 
    Fig.nTileX     = nLead;
    Fig.nTileY     = 2;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Loop over the leads
    for iLead = 1 : numel( leads )

        % Plot Nino 3.4 phases
        set( gcf, 'currentAxes', ax( iLead, 1 ) )
        plotPhaseEvolution( Nino34.idx', selectIndNino34, anglesNino34, ...
                            phase0, leads( iLead ) ) 
        xlabel( 'Nino 3.4' )
        xlim( PlotLim.nino34 )
        ylim( PlotLim.nino34 )
        if iLead > 1 
            yticklabels( [] )
        else
            ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
        end
        title( sprintf( 'Lead = %i months', leads( iLead ) ) )
        
        % Plot NLSA phases 
        set( gcf, 'currentAxes', ax( iLead, 2 ) )
        plotPhaseEvolution( Phi.idx', selectIndPhi, anglesPhi, ...
                            phase0, leads( iLead ) )
        xlabel( sprintf( '\\phi_{%i}', idxPhiEnso( 1 ) ) )
        if iLead > 1
            yticklabels( [] )
        else
            ylabel( sprintf( '\\phi_{%i}', idxPhiEnso( 2 ) ) )
        end
        xlim( [ -3 3 ] )
        ylim( [ -3 3 ] )
    end

    title( axTitle, sprintf( 'Start phase = %i', phase0 ) )

    % Print figure
    if ifPrintFig
        figFile = sprintf( 'figEnsoEquivarianceKernel_phase%i.png', phase0 );
        figFile = fullfile( figDir, figFile );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end

%% EQUIVARIANCE PLOTS BASED ON GENERATOR
if ifKoopmanEquivariance

    nLead = numel( leads );  

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 10; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .1;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .5;
    Fig.gapX       = .20;
    Fig.gapY       = .5;
    Fig.gapT       = .25; 
    Fig.nTileX     = nLead;
    Fig.nTileY     = 2;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Loop over the leads
    for iLead = 1 : numel( leads )

        % Plot Nino 3.4 phases
        set( gcf, 'currentAxes', ax( iLead, 1 ) )
        plotPhaseEvolution( Nino34.idx', selectIndNino34, anglesNino34, ...
                            phase0, leads( iLead ) ) 
        xlabel( 'Nino 3.4' )
        xlim( PlotLim.nino34 )
        ylim( PlotLim.nino34 )
        if iLead > 1 
            yticklabels( [] )
        else
            ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
        end
        title( sprintf( 'Lead = %i months', leads( iLead ) ) )
        
        % Plot Koopman phases 
        set( gcf, 'currentAxes', ax( iLead, 2 ) )
        plotPhaseEvolution( Z.idx', selectIndZ, anglesZ, ...
                            phase0, leads( iLead ) )
        xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
        if iLead > 1
            yticklabels( [] )
        else
            ylabel( sprintf( 'Im(z_{%i})', idxZEnso ) )
        end
        xlim( [ -2.5 2.5 ] )
        ylim( [ -2.5 2.5 ] )
    end

    title( axTitle, sprintf( 'Start phase = %i', phase0 ) )

    % Print figure
    if ifPrintFig
        figFile = sprintf( 'figEnsoEquivarianceGenerator_phase%i.png', phase0);
        figFile = fullfile( figDir, figFile );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end

%% GENERAL PARAMETERS FOR PHASE COMPOSITES

% Parameters that depend on plot domain
switch compositesDomain
case 'globe'
    xLim    = [ 0 359 ]; % longitudde limits
    yLim    = 70;        %  latitude limits are +/- of this value    
    aspectR = ( 3 / 4 ) ^ 3; % axis aspect ratio
case 'tropics'
    xLim    = [ 0 359 ]; % longitudde limits
    yLim    = 30;        %  latitude limits are +/- of this value    
    aspectR = ( 3 / 4 ) ^ 4; % axis aspect ratio
case 'Pacific'
    xLim    = [ 100 290 ]; % longitudde limits
    yLim    = 40;        %  latitude limits are +/- of this value    
    aspectR = ( 3 / 4 ) ^ 3; % axis aspect ratio
otherwise
    error( 'Invalid composites domain' )
end



% Figure and axes parameters 
Fig.units      = 'inches';
Fig.figWidth   = 13; 
Fig.deltaX     = .55;
Fig.deltaX2    = .7;
Fig.deltaY     = .5;
Fig.deltaY2    = .5;
Fig.gapX       = .20;
Fig.gapY       = .2;
Fig.gapT       = .25; 
Fig.nTileX     = 4;
Fig.nTileY     = nPhase;
Fig.aspectR    = aspectR;
Fig.fontName   = 'helvetica';
Fig.fontSize   = 6;
Fig.tickLength = [ 0.02 0 ];
Fig.visible    = 'on';
Fig.nextPlot   = 'add'; 

% SST field
% SST.ifXY is logical mask for valid ocean gridpoints
SST = load( fullfile( model.trgComponent( iCSST ).path, ...
                      'dataGrid.mat' ) ); 
SST.xLim = xLim; % longitude plot limits
SST.yLim = yLim; % latitude plot limits
SST.cLim    = [ -2 2 ]; % color range
SST.cOffset = .035; % horizontal offset of colorbar
SST.cScale  = .4;  % scaling factor for colorbar width  
SST.ifXTickLabels = true;  
SST.ifYTickLabels = true;
    
% SSH field
% SSH.ifXY is logical mask for valid ocean gridpoints
SSH = load( fullfile( model.trgComponent( iCSSH ).path, ...
                      'dataGrid.mat' ) ); 
SSH.xLim = xLim; % longitude plot limits
SSH.yLim = yLim; % latitude plot limits
SSH.cLim    = [ -15 15 ]; % color range
SSH.cOffset = .035; % horizontal offset of colorbar
SSH.cScale  = .4;  % scaling factor for colorbar width  
SSH.ifXTickLabels = true;  
SSH.ifYTickLabels = false;
 

% SAT field
% SAT.ifXY is logical mask for valid gridpoints
SAT = load( fullfile( model.trgComponent( iCSAT ).path, ...
                      'dataGrid.mat' ) ); 
SAT.xLim = xLim; % longitude plot limits
SAT.yLim = yLim; % latitude plot limits
SAT.cLim    = [ -2 2 ]; % color range
SAT.cOffset = .035; % horizontal offset of colorbar
SAT.cScale  = .4;  % scaling factor for colorbar width  
SAT.ifXTickLabels = true;  
SAT.ifYTickLabels = false;
 
% Precipitation rate field
% PRate.ifXY is logical mask for valid gridpoints
PRate = load( fullfile( model.trgComponent( iCPRate ).path, ...
                        'dataGrid.mat' ) ); 
PRate.xLim = xLim; % longitude plot limits
PRate.yLim = yLim; % latitude plot limits
PRate.cLim    = [ -2 2 ]; % color range
PRate.cOffset = .035; % horizontal offset of colorbar
PRate.cScale  = .4;  % scaling factor for colorbar width  
PRate.ifXTickLabels = true;  
PRate.ifYTickLabels = false;
switch dataset
case 'ccsm4Ctrl'
    PRate.scl     = 1000 * 3600 * 24; % convert from m/s to mm/day 
    PRate.title   = 'Precip. anomaly (mm/day)';
end

% Surface wind field
UVWnd = load( fullfile( model.trgComponent( iCUWnd ).path, ...
              'dataGrid.mat' ) ); 
UVWnd.nSkipX = 10; % zonal downsampling factor for quiver plots
UVWnd.nSkipY = 10; % meridional downsampling factor for quiver plots
 

%% COMPOSITES BASED ON NINO 3.4 INDEX
% Create a cell array compPhi of size [ 1 nC ] where nC is the number of 
% observables to be composited. nC is equal to the number of target 
% components in the NLSA model. 
%
% compNino34{ iC } is an array of size [ nD nPhase ], where nD is the dimension
% of component iC. compNino34{ iC }( :, iPhase ) contains the phase 
% composite for observable iC and phase iPhase. 
if ifNinoComposites

    disp( 'Nino 3.4-based composites...' ); t = tic;
    
    % Start and end time indices in data arrays
    iStart = 1 + nSB + nShiftTakens;
    iEnd   = iStart + nSE - 1;  

    compNino34 = computePhaseComposites( model, selectIndNino34, ...
                                         iStart, iEnd );

    toc( t )

    [ fig, ax, axTitle ] = tileAxes( Fig );

    colormap( redblue )


    % Loop over the phases
    for iPhase = 1 : nPhase

        % SST phase composites
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        SST.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compNino34{ iCSST }( :, iPhase ), SST, ...
                                compNino34{ iCUWnd }( :, iPhase ), ...
                                compNino34{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SST anomaly (K), surface wind';
        else
            plotPhaseComposite( compNino34{ iCSST }( :, iPhase ), SST )
            titleStr = 'SST anomaly (K)';
        end
        if iPhase == 1
            title( titleStr )
        end
        lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        lblPos = get( lbl, 'position' );
        lblPos( 1 ) = lblPos( 1 ) - .4;
        set( lbl, 'position', lblPos )

        % SSH phase composites
        set( fig, 'currentAxes', ax( 2, iPhase ) )
        SSH.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compNino34{ iCSSH }( :, iPhase ), SSH, ...
                                compNino34{ iCUWnd }( :, iPhase ), ...
                                compNino34{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SSH anomaly (cm), surface wind';
        else
            plotPhaseComposite( compNino34{ iCSSH }( :, iPhase ), SSH )
            titleStr = 'SSH anomaly (cm)';
        end
        if iPhase == 1
            title( titleStr  )
        end

        % SAT phase composites
        set( fig, 'currentAxes', ax( 3, iPhase ) )
        SAT.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compNino34{ iCSAT }( :, iPhase ), SAT, ...
                                compNino34{ iCUWnd }( :, iPhase ), ...
                                compNino34{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SAT anomaly (K), surface wind';
        else
            plotPhaseComposite( compNino34{ iCSAT }( :, iPhase ), SAT )
            titleStr = 'SAT anomaly (K)';
        end
        if iPhase == 1
            title( titleStr  )
        end

        % Precipitation rate phase composites
        set( fig, 'currentAxes', ax( 4, iPhase ) )
        PRate.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( ...
                compNino34{ iCPRate }( :, iPhase ) * PRate.scl, PRate, ...
                compNino34{ iCUWnd }( :, iPhase ), ...
                compNino34{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = [ PRate.title, ', surface wind' ]; 
        else
            plotPhaseComposite( ...
                compNino34{ iCPRate }( :, iPhase ) * PRate.scl, PRate )
            titleStr = PRate.title;
        end
        if iPhase == 1
            title( titleStr  )
        end
    end

    title( axTitle, 'ENSO composites -- Nino 3.4 index' )

    % Print figure
    if ifPrintFig
        figFile = [ 'figEnsoCompositesNino_' compositesDomain '.png' ];
        figFile = fullfile( figDir, figFile  );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end



%% COMPOSITES BASED ON NLSA
% Create a cell array compPhi of size [ 1 nC ] where nC is the number of 
% observables to be composited. nC is equal to the number of target 
% components in the NLSA model. 
%
% compPhi{ iC } is an array of size [ nD nPhase ], where nD is the dimension
% of component iC. compPhi{ iC }( :, iPhase ) contains the phase composite for 
% observable iC and phase iPhase. 
if ifNLSAComposites

    disp( 'NLSA-based composites...' ); t = tic;
    
    % Start and end time indices in data arrays
    iStart = 1 + nSB + nShiftTakens;
    iEnd   = iStart + nSE - 1;  

    compPhi = computePhaseComposites( model, selectIndPhi, iStart, iEnd );

    toc( t )

    [ fig, ax, axTitle ] = tileAxes( Fig );

    colormap( redblue )

    % Loop over the phases
    for iPhase = 1 : nPhase

        % SST phase composites
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        SST.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compPhi{ iCSST }( :, iPhase ), SST, ...
                                compPhi{ iCUWnd }( :, iPhase ), ...
                                compPhi{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SST anomaly (K), surface wind';
        else
            plotPhaseComposite( compPhi{ iCSST }( :, iPhase ), SST )
            titleStr = 'SST anomaly (K)';
        end
        if iPhase == 1
            title( titleStr  )
        end
        lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        lblPos = get( lbl, 'position' );
        lblPos( 1 ) = lblPos( 1 ) - .4;
        set( lbl, 'position', lblPos )

        % SSH phase composites
        set( fig, 'currentAxes', ax( 2, iPhase ) )
        SSH.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compPhi{ iCSSH }( :, iPhase ), SSH, ...
                                compPhi{ iCUWnd }( :, iPhase ), ...
                                compPhi{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SSH anomaly (K), surface wind';
        else
            plotPhaseComposite( compPhi{ iCSSH }( :, iPhase ), SSH );
            titleStr = 'SSH anomaly (cm)';
        end
        if iPhase == 1
            title( titleStr  )
        end



        % SAT phase composites
        set( fig, 'currentAxes', ax( 3, iPhase ) )
        SAT.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compPhi{ iCSAT }( :, iPhase ), SAT, ...
                                compPhi{ iCUWnd }( :, iPhase ), ...
                                compPhi{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SAT anomaly (K), surface wind';
        else
            plotPhaseComposite( compPhi{ iCSAT }( :, iPhase ), SAT );
            titleStr = 'SAT anomaly (K)';
        end
        if iPhase == 1
            title( titleStr  )
        end

        % Precipitation rate phase composites
        set( fig, 'currentAxes', ax( 4, iPhase ) )
        PRate.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( ...
                compPhi{ iCPRate }( :, iPhase ) * PRate.scl, PRate, ...
                compPhi{ iCUWnd }( :, iPhase ), ...
                compPhi{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = [ PRate.title, ', surface wind' ]; 
        else
            plotPhaseComposite( ...
                compPhi{ iCPRate }( :, iPhase ) * PRate.scl, PRate )
            titleStr = PRate.title;
        end
        if iPhase == 1
            title( titleStr  )
        end
    end

    title( axTitle, 'ENSO composites -- kernel integral operator' )

    % Print figure
    if ifPrintFig
        figFile = [ 'figEnsoCompositesKernel_' compositesDomain '.png' ];
        figFile = fullfile( figDir, figFile  );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end


%% COMPOSITES BASED ON GENERATOR
% Create a cell array compZ of size [ 1 nC ] where nC is the number of 
% observables to be composited. nC is equal to the number of target 
% components in the NLSA model. 
%
% compZ{ iC } is an array of size [ nD nPhase ], where nD is the dimension
% of component iC. compZ{ iC }( :, iPhase ) contains the phase composite for 
% observable iC and phase iPhase. 
if ifKoopmanComposites

    disp( 'Generator-based composites...' ); t = tic;
    
    % Start and end time indices in data arrays
    iStart = 1 + nSB + nShiftTakens;
    iEnd   = iStart + nSE - 1;  

    compZ = computePhaseComposites( model, selectIndZ, iStart, iEnd );

    toc( t )

    [ fig, ax, axTitle ] = tileAxes( Fig );

    colormap( redblue )

    % Loop over the phases
    for iPhase = 1 : nPhase

        % SST phase composites
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        SST.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compZ{ iCSST }( :, iPhase ), SST, ...
                                compZ{ iCUWnd }( :, iPhase ), ...
                                compZ{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SST anomaly (K), surface wind';
        else
            plotPhaseComposite( compZ{ iCSST }( :, iPhase ), SST )
            titleStr = 'SST anomaly (K)';
        end
        if iPhase == 1
            title( titleStr  )
        end
        lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        lblPos = get( lbl, 'position' );
        lblPos( 1 ) = lblPos( 1 ) - .4;
        set( lbl, 'position', lblPos )

        % SSH phase composites
        set( fig, 'currentAxes', ax( 2, iPhase ) )
        SSH.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compZ{ iCSSH }( :, iPhase ), SSH, ...
                                compZ{ iCUWnd }( :, iPhase ), ...
                                compZ{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SSH anomaly (K), surface wind';
        else
            plotPhaseComposite( compZ{ iCSSH }( :, iPhase ), SSH );
            titleStr = 'SSH anomaly (K)';
        end
        if iPhase == 1
            title( titleStr  )
        end

        % SAT phase composites
        set( fig, 'currentAxes', ax( 3, iPhase ) )
        SAT.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( compZ{ iCSAT }( :, iPhase ), SAT, ...
                                compZ{ iCUWnd }( :, iPhase ), ...
                                compZ{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = 'SAT anomaly (K), surface wind';
        else
            plotPhaseComposite( compZ{ iCSAT }( :, iPhase ), SAT );
            titleStr = 'SAT anomaly (K)';
        end
        if iPhase == 1
            title( titleStr  )
        end

        % Precipitation rate phase composites
        set( fig, 'currentAxes', ax( 4, iPhase ) )
        PRate.ifXTickLabels = iPhase == nPhase;
        if ifPlotWind
            plotPhaseComposite( ...
                compZ{ iCPRate }( :, iPhase ) * PRate.scl, PRate, ...
                compZ{ iCUWnd }( :, iPhase ), ... 
                compZ{ iCVWnd }( :, iPhase ), UVWnd )
            titleStr = [ PRate.title, ', surface wind' ]; 
        else
            plotPhaseComposite( ...
                compZ{ iCPRate }( :, iPhase ) * PRate.scl, PRate )
            titleStr = PRate.title;
        end
        if iPhase == 1
            title( titleStr  )
        end
    end

    title( axTitle, 'ENSO composites -- generator' )

    % Print figure
    if ifPrintFig
        figFile = [ 'figEnsoCompositesGenerator_' compositesDomain '.png' ];
        figFile = fullfile( figDir, figFile  );
        print( fig, figFile, '-dpng', '-r300' ) 
    end
end



% AUXILIARY FUNCTIONS

%% Function to plot two-dimensional ENSO index, highlighting significant events
function plotLifecycle( Index, Ninos, Ninas, tFormat )

% plot temporal evolution of index
plot( Index.idx( 1, : ), Index.idx( 2, : ), 'g-' )
hold on
grid on

% highlight significant events
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
          datestr( Index.time( idxT2 ), 'yyyy' ) )
end
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
          datestr( Index.time( idxT2 ), 'yyyy' ) )
end

end

%% Function to plot two-dimensional ENSO index and associated phases
function plotPhases( index, selectInd, angles )

% plot temporal evolution of index
plot( index( :, 1 ), index( :, 2 ), '-', 'Color', [ 1 1 1 ] * .7  )
hold on

% plot phases
nPhase = numel( selectInd );
c = distinguishable_colors( nPhase );
c = c( [ 2 3 4 5 1 6 7 8 ], : );
for iPhase = 1 : nPhase

    plot( index( selectInd{ iPhase }, 1 ), index( selectInd{ iPhase }, 2 ), ...
        '.', 'markersize', 15, 'color', c( iPhase, : ) )
end

end

%% Function to plot ENSO phase evolution
function plotPhaseEvolution( index, selectInd, angles, phase0, lead )

% plot temporal evolution of index
plot( index( :, 1 ), index( :, 2 ), '-', 'Color', [ 1 1 1 ] * .7  )
hold on

% plot phases
nPhase = numel( selectInd );
c = distinguishable_colors( nPhase );
c = c( [ 2 3 4 5 1 6 7 8 ], : );
for iPhase = 1 : nPhase

    plot( index( selectInd{ iPhase }, 1 ), index( selectInd{ iPhase }, 2 ), ...
        '.', 'markersize', 5, 'color', c( iPhase, : ) * .7 )
end

% plot evolution from reference phase
indMax = size( index, 1 );
ind = selectInd{ phase0 } + lead; 
ind = ind( ind <= indMax );
plot( index( ind, 1 ), index( ind, 2 ), ...
    '.', 'markersize', 10, 'color', c( phase0, : ) )   
end

%% Function to compute phase composites from target data of NLSA model
function comp = computePhaseComposites( model, selectInd, iStart, iEnd )

nC = size( model.trgComponent, 1 ); % number of observables to be composited
nPhase = numel( selectInd ); % number of phases       

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
            comp{ iC }( :, iPhase ) = mean( y( :, selectInd{ iPhase } ), 2 );

        end
    end
end

%% Function to plot phase composites
function plotPhaseComposite( s, SGrd, u, v, VGrd )

% s:    values of scalar field to plot
% SGrd: data structure with grid information for scalar field  
% u, v: components of vector field to plot
% VGrd: data structure with grid information for vector field

sData = zeros( size( SGrd.ifXY ) );
sData( ~SGrd.ifXY ) = NaN;
sData( SGrd.ifXY ) = s;

if SGrd.ifXTickLabels
    xTickLabelsArg = { };
else
    xTickLabelsArg = { 'xTickLabels' [] };
end
if SGrd.ifYTickLabels
    yTickLabelsArg = { };
else
    yTickLabelsArg = { 'yTickLabels' [] };
end
m_proj( 'Miller cylindrical', 'long',  SGrd.xLim, 'lat', SGrd.yLim );
if ~isvector( SGrd.lon )
    SGrd.lon = SGrd.lon';
    SGrd.lat = SGrd.lat';
end
h = m_pcolor( SGrd.lon, SGrd.lat, sData' );
set( h, 'edgeColor', 'none' )
m_grid( 'linest', 'none', 'linewidth', 1, 'tickdir', 'out', ...
        xTickLabelsArg{ : }, yTickLabelsArg{ : } ); 
m_coast( 'linewidth', 1, 'color', 'k' );
        %'xTick', [ SGrd.xLim( 1 ) : 40 : SGrd.xLim( 2 ) ], ...
        %'yTick', [ SGrd.yLim( 1 ) : 20 : SGrd.yLim( 2 ) ] );

axPos = get( gca, 'position' );
hC = colorbar( 'location', 'eastOutside' );
cPos   = get( hC, 'position' );
cPos( 1 ) = cPos( 1 ) + SGrd.cOffset;
cPos( 3 ) = cPos( 3 ) * SGrd.cScale;
set( gca, 'cLim', SGrd.cLim, 'position', axPos )
set( hC, 'position', cPos )

if nargin == 2
    return
end

uData = zeros( size( VGrd.ifXY ) );
uData( ~VGrd.ifXY ) = NaN;
uData( VGrd.ifXY ) = u;

vData = zeros( size( VGrd.ifXY ) );
vData( ~VGrd.ifXY ) = NaN;
vData( VGrd.ifXY ) = v;

[ lon, lat ] = meshgrid( VGrd.lon, VGrd.lat );
%size(VGrd.lon)
%size(uData')
%size(vData')
m_quiver( lon( 1 : VGrd.nSkipY : end, 1 : VGrd.nSkipX : end ), ...
          lat( 1 : VGrd.nSkipY : end, 1 : VGrd.nSkipX : end ), ...
          uData( 1 : VGrd.nSkipX : end, 1 : VGrd.nSkipY : end )', ...
          vData( 1 : VGrd.nSkipX : end, 1 : VGrd.nSkipY : end )', 'g-' ) 
end
