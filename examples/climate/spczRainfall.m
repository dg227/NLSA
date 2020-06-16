% ANALYSIS OF SPCZ RAINFALL
%
% Modified 2020/06/16

%% DATA SPECIFICATION 
%dataset    = 'ccsm4Ctrl';
%period     = '1300yr'; 
dataset    = 'cmap';
period     = 'satellite'; 
%sourceVar = 'IPPrecip';   % Indo-Pacific precipitation
sourceVar = 'PacPrecip'; % Pacific precipitation
embWindow  = '4yr';       % 4-year embedding
kernel     = 'cone';       % cone kernel      


%% SCRIPT EXECUTION OPTIONS
% Data extraction
ifDataSource = true;  % extract source precipitation data from NetCDF files  

% ENSO representations
ifNLSA    = true; % compute kernel (NLSA) eigenfunctions
ifKoopman = true; % compute Koopman eigenfunctions

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes


%% EXTRACT SOURCE DATA
if ifDataSource
    disp( sprintf( 'Reading source data %s...', sourceVar ) ); t = tic;
    spczRainfall_data( dataset, period, sourceVar ) 
    toc( t )
end

%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES

experiment = { dataset period sourceVar [ embWindow 'Emb' ] ...
               [ kernel 'Kernel' ] };
experiment = strjoin_e( experiment, '_' );

disp( 'Building NLSA model...' ); t = tic;
model = spczRainfall_nlsaModel( experiment );
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

