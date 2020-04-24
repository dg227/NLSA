% RECONSTRUCT THE LIFECYCLE OF THE EL NINO SOUTHERN OSCILLATION (ENSO) 
% USING DATA-DRIVEN SPECTRAL ANALYSIS OF KOOPMAN/TRANSFER OPERATORS
%
% Modified 2020/04/20

%% DATA SPECIFICATION AND GLOBAL PARAMETERS 
dataset    = 'noaa';           % NOAA 20th Century Reanalysis v2
experiment = 'enso_lifecycle'; % data analysis experiment 

nShiftNino   = 11;        % temporal shift to obtain 2D Nino index
idxPhiEnso   = [ 10 9 ];  % ENSO eigenfunctions from NLSA (kernel operator)
signPhi      = [ -1 -1 ]; % multiplication factor (for consistency with Nino)
idxZEnso     = 9;         % ENSO eigenfunction from generator      
signZ        = -1;        % multiplication factor (for consistency with Nino)
nPhase       = 8;         % number of ENSO phases
nSamplePhase = 100;       % number of samples per phase
phase0       = 5;         % start phase in equivariance plots
leads        = [ 0 6 12 18 24 ]; % leads (in months) for equivariance plots

%% EL NINO/LA NINA EVENTS
% El Nino/La Nina events to mark up in lifecycle plots (in yyyymm format)
ElNinos = { { '201511' '201603' } ... 
            { '199711' '199803' } ...
            { '199111' '199203' } ...
            { '198711' '198803' } ...
            { '198211' '198303' } };

LaNinas = { { '201011' '201103' } ... 
            { '200711' '200803' } ...
            { '199911' '200003' } ...
            { '199811' '199903' } ...
            { '198811' '198903' } };


%% BATCH PROCCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% SCRIPT EXECUTION OPTIONS

% Data extraction
ifDataSST    = false; % extract SST data from NetCDF source files
ifDataSAT    = false; % extract SAT data from NetCDF source files
ifDataPrecip = false; % extract precipitation data from NetCDF source files  
ifDataWind   = false;  % extract 10m wind data from NetCDF source files  

% ENSO representations
ifNLSA    = false; % compute kernel (NLSA) eigenfunctions
ifKoopman = false; % compute Koopman eigenfunctions
ifNinoIdx = false;  % compute two-dimensional (lead/lag) Nino 3.4 index  

% ENSO 2D lifecycle plots
ifNLSALifecycle    = false; % plot ENSO lifecycle from kernel eigenfunctions
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

% Output options
ifPrintFig = true; % print figures to file

%% EXTRACT SST DATA
if ifDataSST
    % Create data structure with input data specifications, and retrieve 
    % input data. Data is saved on disk. 
 
    % Input data directory 
    dirName    = '/Volumes/TooMuch/physics/climate/data/noaa'; 

    % Time specification
    DataSpecs.Time.tLim    = { '187001' '201906' }; % time limits
    DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology time limits
    DataSpecs.Time.tStart  = '185401';              % start time in nc file 
    DataSpecs.Time.tFormat = 'yyyymm';              % time format

    % File name for SST field
    fileName   = 'sst.mnmean.v4-4.nc'; % filename base for input data
    varName    = 'sst';                % variable name in NetCDF file 

    % Input data
    DataSpecs.In.dir  = dirName;
    DataSpecs.In.file = fileName;
    DataSpecs.In.var  = varName;
    
    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );
    DataSpecs.Out.fld = varName;      

    % Read area-weighted SST data for Indo-Pacific domain
    disp( 'Reading Indo-Pacific SST data...' ); t = tic;

    DataSpecs.Domain.xLim = [ 28 290 ]; % longitude limits
    DataSpecs.Domain.yLim = [ -60 20 ]; % latitude limits
    
    DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
    DataSpecs.Opts.ifWeight      = true;  % perform area weighting
    DataSpecs.Opts.ifCenterMonth = false; % don't remove monthly climatology 
    DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
    DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
    DataSpecs.Opts.ifWrite       = true;  % write data to disk

    climateData( dataset, DataSpecs ) % read SST data
    toc( t )

    % Read Nino 3.4 index 
    disp( 'Reading Nino 3.4 data...' ); t = tic; 

    DataSpecs.Domain.xLim = [ 190 240 ]; % longitude limits 
    DataSpecs.Domain.yLim = [ -5 5 ];    % latitude limits

    DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
    DataSpecs.Opts.ifWeight      = true;  % perform area weighting
    DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
    DataSpecs.Opts.ifAverage     = true;  % perform area averaging
    DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
    DataSpecs.Opts.ifWrite       = true;  % write data to disk

    climateData( dataset, DataSpecs ) % read Nino 3.4 data
    toc( t )

    % Read global SST
    disp( 'Reading global SST data...' ); t = tic; 

    DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
    DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

    DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
    DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
    DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
    DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
    DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
    DataSpecs.Opts.ifWrite       = true;  % write data to disk

    climateData( dataset, DataSpecs ) % read global SST data
    toc( t )
end

%% EXTRACT SAT DATA
if ifDataSAT
    % Create data structure with input data specifications, and retrieve 
    % input data. Data is saved on disk. 
 
    % Input data directory 
    dirName    = '/Volumes/TooMuch/physics/climate/data/noaa'; 

    % Time specification
    DataSpecs.Time.tLim    = { '187001' '201906' }; % time limits
    DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology time limits
    DataSpecs.Time.tStart  = '188001';              % start time in nc file 
    DataSpecs.Time.tFormat = 'yyyymm';              % time format

    % File name for SAT field
    fileName   = 'air.mon.anom.v4.nc'; % filename base for input data
    varName    = 'air';                % variable name in NetCDF file 

    % Input data
    DataSpecs.In.dir  = dirName;
    DataSpecs.In.file = fileName;
    DataSpecs.In.var  = varName;
    
    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );
    DataSpecs.Out.fld = varName;      

    % Read global SAT
    disp( 'Reading global SAT (air) data...' ); t = tic; 

    DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
    DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

    DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
    DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
    DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
    DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
    DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
    DataSpecs.Opts.ifWrite       = true;  % write data to disk

    climateData( dataset, DataSpecs ) % read global SAT data
    toc( t )
end

%% EXTRACT PRECIPITATION RATE DATA
if ifDataPrecip
    % Create data structure with input data specifications, and retrieve 
    % input data. Data is saved on disk. 
 
    % Input data directory 
    dirName    = '/Volumes/TooMuch/physics/climate/data/noaa'; 

    % Time specification
    DataSpecs.Time.tLim    = { '187001' '201906' }; % time limits
    DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology time limits
    DataSpecs.Time.tStart  = '185101';              % start time in nc file 
    DataSpecs.Time.tFormat = 'yyyymm';              % time format

    % File name for precipitation field
    fileName   = 'prate.mon.mean.nc'; % filename base for input data
    varName    = 'prate';             % variable name in NetCDF file 

    % Input data
    DataSpecs.In.dir  = dirName;
    DataSpecs.In.file = fileName;
    DataSpecs.In.var  = varName;
    
    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );
    DataSpecs.Out.fld = varName;      

    % Read global precipitation
    disp( 'Reading global precipitation rate data...' ); t = tic; 

    DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
    DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

    DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
    DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
    DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
    DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
    DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
    DataSpecs.Opts.ifWrite       = true;  % write data to disk

    climateData( dataset, DataSpecs ) % read global SAT data
    toc( t )
end

%% EXTRACT SURFACE WIND DATA
if ifDataWind
    % Create data structure with input data specifications, and retrieve 
    % input data. Data is saved on disk. 
 
    % Input/output data directory 
    dirName    = '/Volumes/TooMuch/physics/climate/data/noaa'; 
    DataSpecs.In.dir  = dirName;
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );

    % Time specification
    DataSpecs.Time.tLim    = { '187001' '201906' }; % time limits
    DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology time limits
    DataSpecs.Time.tStart  = '185101';              % start time in nc file 
    DataSpecs.Time.tFormat = 'yyyymm';              % time format

    % Domain specification 
    DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
    DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

    % Output options
    DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
    DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
    DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
    DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
    DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
    DataSpecs.Opts.ifWrite       = true;  % write data to disk

    % File name for zonal wind field field
    fileName   = 'uwnd.10m.mon.mean.nc'; % filename base for input data
    varName    = 'uwnd';                 % variable name in NetCDF file 

    % Input/output names 
    DataSpecs.In.file = fileName;
    DataSpecs.In.var  = varName;
    DataSpecs.Out.fld = varName;      

    % Read global zonal winds
    disp( 'Reading global zonal wind data...' ); t = tic; 

    climateData( dataset, DataSpecs ) 
    toc( t )

    % File name for meridional wind field field
    fileName   = 'vwnd.10m.mon.mean.nc'; % filename base for input data
    varName    = 'vwnd';                 % variable name in NetCDF file 

    % Input/output names 
    DataSpecs.In.file = fileName;
    DataSpecs.In.var  = varName;
    DataSpecs.Out.fld = varName;      

    % Read global meridional winds
    disp( 'Reading global meridional wind data...' ); t = tic; 

    climateData( dataset, DataSpecs ) 
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
[ model, In ] = climateNLSAModel( dataset, experiment ); 

nSE          = getNTotalSample( model.embComponent );
nSB          = getNXB( model.embComponent );
nShiftTakens = floor( getEmbeddingWindow( model.embComponent ) / 2 );
toc( t )

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
% Build a data structure Nino such that:
% 
% Nino.idx is an array of size [ 2 nSE ], where nSE is the number of samples 
% after delay embedding. Nino.idx( 1, : ) contains the values of the Nino 3.4 
% index at the current time. Nino( 2, : ) contains the values of the Nino 3.4 
% index at nShiftNino timesteps (months) in the past.
% 
% Nino.time is an array of size [ 1 nSE ] containing the timestamps in
% Matlab serial date number format. 
if ifNinoIdx

    disp( 'Constructing lagged Nino 3.4 index...' ); t = tic;

    % Timestamps
    Nino.time = getTrgTime( model ); 
    Nino.time = Nino.time( nSB + 1 + nShiftTakens : end );
    Nino.time = Nino.time( 1 : nSE );

    % Nino 3.4 index
    nino = getData( model.trgComponent( 1 ) );
    Nino.idx = [ nino( nShiftNino + 1 : end ) 
                 nino( 1 : end - nShiftNino ) ];
    Nino.idx = Nino.idx( :, nSB + nShiftTakens - nShiftNino + 1 : end );
    Nino.idx = Nino.idx( :, 1 : nSE );
end

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

    % Plot Nino lifecycle
    set( gcf, 'currentAxes', ax( 1 ) )
    plotLifecycle( Nino, ElNinos, LaNinas, model.tFormat )
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )

    % Plot NLSA lifecycle
    set( gcf, 'currentAxes', ax( 2 ) )
    plotLifecycle( Phi, ElNinos, LaNinas, model.tFormat )
    xlabel( sprintf( '\\phi_{%i}', idxPhiEnso( 1 ) ) )
    ylabel( sprintf( '\\phi_{%i}', idxPhiEnso( 2 ) ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    title( 'Kernel integral operator' )

    % Print figure
    if ifPrintFig
        figFile = 'figEnsoLifecycleKernel.png';
        print( figFile, '-dpng', '-r300' ) 
    end
end

%% PLOT ENSO LIFECYCLE BASED ON KOOPMAN EIGENFUNCTIONS
if ifKoopmanLifecycle

    % Retrieve Koopman eigenfunctions
    z = getKoopmanEigenfunctions( model );
    T = getEigenperiods( model.koopmanOp );
    TEnso = T( idxZEnso ) / 12;
    Z.idx = signZ' .*  [ real( z( :, idxZEnso ) )' 
                         imag( z( :, idxZEnso ) )' ];
    Z.time = getTrgTime( model );
    Z.time = Z.time( nSB + 1 + nShiftTakens : end );
    Z.time = Z.time( 1 : nSE );
    
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

    % Plot Nino lifecycle
    set( gcf, 'currentAxes', ax( 1 ) )
    plotLifecycle( Nino, ElNinos, LaNinas, model.tFormat )
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )

    % Plot generator lifecycle
    set( gcf, 'currentAxes', ax( 2 ) )
    plotLifecycle( Z, ElNinos, LaNinas, model.tFormat )
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    ylabel( sprintf( 'Im(z_{%i})', idxZEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    title( sprintf( 'Generator; eigenperiod = %1.2f y', TEnso ) )

    % Print figure
    if ifPrintFig
        figFile = 'figEnsoLifecycleGenerator.png';
        print( figFile, '-dpng', '-r300' ) 
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
% avNinoIndPhi is a row vector of size [ 1 nPhase ] containing the average
% Nino 3.4 index for each NLSA phase. 
%
% selectIndNino, anglesNino, and avNinoIndNino are defined analogously to
% selectIndPhi, anglesPhi, and avNinoIndPhi, respectively, using the Nino 3.4
% index. 
if ifNLSAPhases
   
    % Compute ENSO phases based on NLSA
    [ selectIndPhi, anglesPhi, avNinoIndPhi ] = computeLifecyclePhases( ...
        Phi.idx', Nino.idx( 1, : )', nPhase, nSamplePhase );

    % Compute ENSO phases based on Nino 3.4 index
    [ selectIndNino, anglesNino, avNinoIndNino ] = computeLifecyclePhases( ...
        Nino.idx', Nino.idx(1,:)', nPhase, nSamplePhase );
        
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
    plotPhases( Nino.idx', selectIndNino, anglesNino ) 
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )

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
        figFile = 'figEnsoPhasesKernel.png';
        print( figFile, '-dpng', '-r300' ) 
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
% avNinoIndZ is a row vector of size [ 1 nPhase ] containing the average
% Nino 3.4 index for each NLSA generator. 
%
% selectIndNino, anglesNino, and avNinoIndNino are defined analogously to
% selectIndZ, anglesZ, and avNinoIndZ, respectively, using the Nino 3.4
% index. 
if ifKoopmanPhases
   
    % Compute ENSO phases based on generator
    [ selectIndZ, anglesZ, avNinoIndZ ] = computeLifecyclePhases( ...
        Z.idx', Nino.idx( 1, : )', nPhase, nSamplePhase );

    % Compute ENSO phases based on Nino 3.4 index
    [ selectIndNino, anglesNino, avNinoIndNino ] = computeLifecyclePhases( ...
        Nino.idx', Nino.idx(1,:)', nPhase, nSamplePhase );
        
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
    plotPhases( Nino.idx', selectIndNino, anglesNino ) 
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )

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
        figFile = 'figEnsoPhasesKoopman.png';
        print( figFile, '-dpng', '-r300' ) 
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
        plotPhaseEvolution( Nino.idx', selectIndNino, anglesNino, ...
                            phase0, leads( iLead ) ) 
        xlabel( 'Nino 3.4' )
        xlim( [ -3 3 ] )
        ylim( [ -3 3 ] )
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
        print( figFile, '-dpng', '-r300' ) 
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
        plotPhaseEvolution( Nino.idx', selectIndNino, anglesNino, ...
                            phase0, leads( iLead ) ) 
        xlabel( 'Nino 3.4' )
        xlim( [ -3 3 ] )
        ylim( [ -3 3 ] )
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
        print( figFile, '-dpng', '-r300' ) 
    end
end

%% COMPOSITES BASED ON NINO 3.4 INDEX
% Create a cell array compPhi of size [ 1 nC ] where nC is the number of 
% observables to be composited. nC is equal to the number of target 
% components in the NLSA model. 
%
% compNino{ iC } is an array of size [ nD nPhase ], where nD is the dimension
% of component iC. compNino{ iC }( :, iPhase ) contains the phase 
% composite for observable iC and phase iPhase. 
if ifNinoComposites

    disp( 'Nino 3.4-based composites...' ); t = tic;
    
    % Start and end time indices in data arrays
    iStart = 1 + nSB + nShiftTakens;
    iEnd   = iStart + nSE - 1;  

    compNino = computePhaseComposites( model, selectIndNino, iStart, iEnd );

    toc( t )

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 10; 
    Fig.deltaX     = .55;
    Fig.deltaX2    = .7;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .5;
    Fig.gapX       = .20;
    Fig.gapY       = .2;
    Fig.gapT       = .25; 
    Fig.nTileX     = 3;
    Fig.nTileY     = nPhase;
    Fig.aspectR    = ( 3 / 4 ) ^ 3;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    colormap( redblue )

    % Retrieve grid data for SST field
    % SST.ifXY is logical mask for valid ocean gridpoints
    SST = load( fullfile( model.trgComponent( 2 ).path, 'dataGrid.mat' ) ); 
    SST.xLim = [ min( SST.lon ) max( SST.lon ) ]; % longitude plot limits
    SST.yLim = [ min( SST.lat ) max( SST.lat ) ]; % latitude plot limits
    SST.cLim    = [ -2 2 ]; % color range
    SST.cOffset = .05; % horizontal offset of colorbar
    SST.cScale  = .4;  % scaling factor for colorbar width  
    SST.ifXTickLabels = true;  
    SST.ifYTickLabels = true;
    
    % Retrieve grid data for SAT field
    % SAT.ifXY is logical mask for valid gridpoints
    SAT = load( fullfile( model.trgComponent( 3 ).path, 'dataGrid.mat' ) ); 
    SAT.xLim = [ min( SAT.lon ) max( SAT.lon ) ]; % longitude plot limits
    SAT.yLim = [ min( SAT.lat ) max( SAT.lat ) ]; % latitude plot limits
    SAT.cLim    = [ -2 2 ]; % color range
    SAT.cOffset = .05; % horizontal offset of colorbar
    SAT.cScale  = .4;  % scaling factor for colorbar width  
    SAT.ifXTickLabels = true;  
    SAT.ifYTickLabels = false;
 
    % Retrieve grid data for precipitation rate field
    % PRate.ifXY is logical mask for valid gridpoints
    PRate = load( fullfile( model.trgComponent( 4 ).path, 'dataGrid.mat' ) ); 
    PRate.xLim = [ min( PRate.lon ) max( PRate.lon ) ]; % longitude plot limits
    PRate.yLim = [ min( PRate.lat ) max( PRate.lat ) ]; % latitude plot limits
    PRate.cLim    = [ -2 2 ]; % color range
    PRate.cOffset = .05; % horizontal offset of colorbar
    PRate.cScale  = .4;  % scaling factor for colorbar width  
    PRate.ifXTickLabels = true;  
    PRate.ifYTickLabels = false;
 
    % Retrieve grid data for surface wind field
    UVWnd = load( fullfile( model.trgComponent( 4 ).path, 'dataGrid.mat' ) ); 
    UVWnd.nSkipX = 5;
    UVWnd.nSkipY = 5;
 
    % Loop over the phases
    for iPhase = 1 : nPhase

        % SST phase composites
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        if iPhase == 1
            title( 'SST anomaly (K), surface wind' )
        end
        SST.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compNino{ 2 }( :, iPhase ), SST, ...
            compNino{ 5 }( :, iPhase ), compNino{ 6 }( :, iPhase ), UVWnd )
        lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        lblPos = get( lbl, 'position' );
        lblPos( 1 ) = lblPos( 1 ) - .4;
        set( lbl, 'position', lblPos )

        % SAT phase composites
        set( fig, 'currentAxes', ax( 2, iPhase ) )
        if iPhase == 1
            title( 'SAT anomaly (K), surface wind' )
        end
        SAT.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compNino{ 3 }( :, iPhase ), SAT, ...
            compNino{ 5 }( :, iPhase ), compNino{ 6 }( :, iPhase ), UVWnd )

        % Precipitation rate phase composites
        set( fig, 'currentAxes', ax( 3, iPhase ) )
        if iPhase == 1
            title( 'PRate anomaly (cg/m^2/s), surface wind' )
        end
        PRate.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compNino{ 4 }( :, iPhase ) * 1E5, PRate, ...
            compNino{ 5 }( :, iPhase ), compNino{ 6 }( :, iPhase ), UVWnd )
    end




    title( axTitle, 'ENSO composites -- Nino 3.4 index' )


    % Print figure
    if ifPrintFig
        figFile = 'figEnsoCompositesNino.png';
        print( figFile, '-dpng', '-r300' ) 
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

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 10; 
    Fig.deltaX     = .55;
    Fig.deltaX2    = .7;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .5;
    Fig.gapX       = .20;
    Fig.gapY       = .2;
    Fig.gapT       = .25; 
    Fig.nTileX     = 3;
    Fig.nTileY     = nPhase;
    Fig.aspectR    = ( 3 / 4 ) ^ 3;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    colormap( redblue )

    % Retrieve grid data for SST field
    % SST.ifXY is logical mask for valid ocean gridpoints
    SST = load( fullfile( model.trgComponent( 2 ).path, 'dataGrid.mat' ) ); 
    SST.xLim = [ min( SST.lon ) max( SST.lon ) ]; % longitude plot limits
    SST.yLim = [ min( SST.lat ) max( SST.lat ) ]; % latitude plot limits
    SST.cLim    = [ -2 2 ]; % color range
    SST.cOffset = .05; % horizontal offset of colorbar
    SST.cScale  = .4;  % scaling factor for colorbar width  
    SST.ifXTickLabels = true;  
    SST.ifYTickLabels = true;
    
    % Retrieve grid data for SAT field
    % SAT.ifXY is logical mask for valid gridpoints
    SAT = load( fullfile( model.trgComponent( 3 ).path, 'dataGrid.mat' ) ); 
    SAT.xLim = [ min( SAT.lon ) max( SAT.lon ) ]; % longitude plot limits
    SAT.yLim = [ min( SAT.lat ) max( SAT.lat ) ]; % latitude plot limits
    SAT.cLim    = [ -2 2 ]; % color range
    SAT.cOffset = .05; % horizontal offset of colorbar
    SAT.cScale  = .4;  % scaling factor for colorbar width  
    SAT.ifXTickLabels = true;  
    SAT.ifYTickLabels = false;
 
    % Retrieve grid data for precipitation rate field
    % PRate.ifXY is logical mask for valid gridpoints
    PRate = load( fullfile( model.trgComponent( 4 ).path, 'dataGrid.mat' ) ); 
    PRate.xLim = [ min( PRate.lon ) max( PRate.lon ) ]; % longitude plot limits
    PRate.yLim = [ min( PRate.lat ) max( PRate.lat ) ]; % latitude plot limits
    PRate.cLim    = [ -2 2 ]; % color range
    PRate.cOffset = .05; % horizontal offset of colorbar
    PRate.cScale  = .4;  % scaling factor for colorbar width  
    PRate.ifXTickLabels = true;  
    PRate.ifYTickLabels = false;
 
    % Retrieve grid data for surface wind field
    UVWnd = load( fullfile( model.trgComponent( 4 ).path, 'dataGrid.mat' ) ); 
    UVWnd.nSkipX = 5;
    UVWnd.nSkipY = 5;
 
    % Loop over the phases
    for iPhase = 1 : nPhase

        % SST phase composites
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        if iPhase == 1
            title( 'SST anomaly (K), surface wind' )
        end
        SST.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compPhi{ 2 }( :, iPhase ), SST, ...
            compPhi{ 5 }( :, iPhase ), compPhi{ 6 }( :, iPhase ), UVWnd )
        lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        lblPos = get( lbl, 'position' );
        lblPos( 1 ) = lblPos( 1 ) - .4;
        set( lbl, 'position', lblPos )

        % SAT phase composites
        set( fig, 'currentAxes', ax( 2, iPhase ) )
        if iPhase == 1
            title( 'SAT anomaly (K), surface wind' )
        end
        SAT.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compPhi{ 3 }( :, iPhase ), SAT, ...
            compPhi{ 5 }( :, iPhase ), compPhi{ 6 }( :, iPhase ), UVWnd )

        % Precipitation rate phase composites
        set( fig, 'currentAxes', ax( 3, iPhase ) )
        if iPhase == 1
            title( 'PRate anomaly (cg/m^2/s), surface wind' )
        end
        PRate.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compPhi{ 4 }( :, iPhase ) * 1E5, PRate, ...
            compPhi{ 5 }( :, iPhase ), compPhi{ 6 }( :, iPhase ), UVWnd )
    end

    title( axTitle, 'ENSO composites -- kernel integral operator' )

    % Print figure
    if ifPrintFig
        figFile = 'figEnsoCompositesKernel.png';
        print( figFile, '-dpng', '-r300' ) 
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

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 10; 
    Fig.deltaX     = .55;
    Fig.deltaX2    = .7;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .5;
    Fig.gapX       = .20;
    Fig.gapY       = .2;
    Fig.gapT       = .25; 
    Fig.nTileX     = 3;
    Fig.nTileY     = nPhase;
    Fig.aspectR    = ( 3 / 4 ) ^ 3;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    colormap( redblue )

    % Retrieve grid data for SST field
    % SST.ifXY is logical mask for valid ocean gridpoints
    SST = load( fullfile( model.trgComponent( 2 ).path, 'dataGrid.mat' ) ); 
    SST.xLim = [ min( SST.lon ) max( SST.lon ) ]; % longitude plot limits
    SST.yLim = [ min( SST.lat ) max( SST.lat ) ]; % latitude plot limits
    SST.cLim    = [ -2 2 ]; % color range
    SST.cOffset = .05; % horizontal offset of colorbar
    SST.cScale  = .4;  % scaling factor for colorbar width  
    SST.ifXTickLabels = true;  
    SST.ifYTickLabels = true;
    
    % Retrieve grid data for SAT field
    % SAT.ifXY is logical mask for valid gridpoints
    SAT = load( fullfile( model.trgComponent( 3 ).path, 'dataGrid.mat' ) ); 
    SAT.xLim = [ min( SAT.lon ) max( SAT.lon ) ]; % longitude plot limits
    SAT.yLim = [ min( SAT.lat ) max( SAT.lat ) ]; % latitude plot limits
    SAT.cLim    = [ -2 2 ]; % color range
    SAT.cOffset = .05; % horizontal offset of colorbar
    SAT.cScale  = .4;  % scaling factor for colorbar width  
    SAT.ifXTickLabels = true;  
    SAT.ifYTickLabels = false;
 
    % Retrieve grid data for precipitation rate field
    % PRate.ifXY is logical mask for valid gridpoints
    PRate = load( fullfile( model.trgComponent( 4 ).path, 'dataGrid.mat' ) ); 
    PRate.xLim = [ min( PRate.lon ) max( PRate.lon ) ]; % longitude plot limits
    PRate.yLim = [ min( PRate.lat ) max( PRate.lat ) ]; % latitude plot limits
    PRate.cLim    = [ -2 2 ]; % color range
    PRate.cOffset = .05; % horizontal offset of colorbar
    PRate.cScale  = .4;  % scaling factor for colorbar width  
    PRate.ifXTickLabels = true;  
    PRate.ifYTickLabels = false;
 
    % Retrieve grid data for surface wind field
    UVWnd = load( fullfile( model.trgComponent( 4 ).path, 'dataGrid.mat' ) ); 
    UVWnd.nSkipX = 5;
    UVWnd.nSkipY = 5;
 
    % Loop over the phases
    for iPhase = 1 : nPhase

        % SST phase composites
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        if iPhase == 1
            title( 'SST anomaly (K), surface wind' )
        end
        SST.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compZ{ 2 }( :, iPhase ), SST, ...
            compZ{ 5 }( :, iPhase ), compZ{ 6 }( :, iPhase ), UVWnd )
        lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        lblPos = get( lbl, 'position' );
        lblPos( 1 ) = lblPos( 1 ) - .4;
        set( lbl, 'position', lblPos )

        % SAT phase composites
        set( fig, 'currentAxes', ax( 2, iPhase ) )
        if iPhase == 1
            title( 'SAT anomaly (K), surface wind' )
        end
        SAT.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compZ{ 3 }( :, iPhase ), SAT, ...
            compZ{ 5 }( :, iPhase ), compZ{ 6 }( :, iPhase ), UVWnd )

        % Precipitation rate phase composites
        set( fig, 'currentAxes', ax( 3, iPhase ) )
        if iPhase == 1
            title( 'PRate anomaly (cg/m^2/s), surface wind' )
        end
        PRate.ifXTickLabels = iPhase == nPhase;
        plotPhaseComposite( compZ{ 4 }( :, iPhase ) * 1E5, PRate, ...
            compZ{ 5 }( :, iPhase ), compZ{ 6 }( :, iPhase ), UVWnd )
    end





    title( axTitle, 'ENSO composites -- generator' )

    % Print figure
    if ifPrintFig
        figFile = 'figEnsoCompositesGenerator.png';
        print( figFile, '-dpng', '-r300' ) 
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
m_proj( 'Miller cylindrical', 'lat',  70, 'long', [ 0 359 ] );
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
