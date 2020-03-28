% RECONSTRUCT THE LIFECYCLE OF THE EL NINO SOUTHERN OSCILLATION (ENSO) 
% USING DATA-DRIVEN KOOPMAN SPECTRAL ANALYSIS
%
% Modified 2020/03/28

%% DATA SPECIFICATION 
dataset    = 'noaa';           % NOAA 20th Century Reanalysis v2
experiment = 'enso_lifecycle'; % data analysis experiment 
dirName    = '/Volumes/TooMuch/physics/climate/data/noaa'; % input data dir.
fileName   = 'sst.mnmean.v4-4.nc'; % filename base for input data
varName    = 'sst';                % variable name in NetCDF file 

%% BATCH PROCCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% SCRIPT EXECUTION OPTIONS
ifData    = true; % extract data from NetCDF source files
ifNLSA    = true; % compute kernel (NLSA) eigenfunctions
ifKoopman = true; % compute Koopman eigenfunctions


%% EXTRACT DATA
if ifData
    % Create data structure with input data specifications, and retrieve 
    % input data. Data is saved on disk. 
 
    % Input data
    DataSpecs.In.dir  = dirName;
    DataSpecs.In.file = fileName;
    DataSpecs.In.var  = varName;
    
    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );
    DataSpecs.Out.fld = varName;      

    % Time specification
    DataSpecs.Time.tLim    = { '187001' '201906' }; % time limits
    DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology time limits
    DataSpecs.Time.tStart  = '185401';              % start time in nc file 
    DataSpecs.Time.tFormat = 'yyyymm';              % time format

    % Read SST data for Indo-Pacific domain
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
end

%% PERFORM NLSA
if ifNLSA
    
    % Build NLSA model. In is a data structure containing the NLSA parameters
    % for the training data
    disp( 'Building NLSA model...' ); t = tic;
    [ model, In ] = climateNLSAModel( dataset, experiment ); 
    toc( t )

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
