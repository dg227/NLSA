% ANALYSIS OF SPCZ RAINFALL
%
% Modified 2020/05/08

%% DATA SPECIFICATION 
dataset    = 'ccsm4Ctrl';
period     = '200yr'; 
experiment = 'IPprecip'; % Indo-Pacific precipitation


%% SCRIPT EXECUTION OPTIONS
% Data extraction
ifDataPrecip = true;  % extract precipitation data from NetCDF source files  

% ENSO representations
ifNLSA    = false; % compute kernel (NLSA) eigenfunctions
ifKoopman = false; % compute Koopman eigenfunctions

%% EXTRACT PRECIPITATION RATE DATA
if ifDataPrecip

    disp( 'Reading precipitation rate data...' ); t = tic; 
    spczRainfall_data( dataset, period, 'IPprecip' )
    toc( t )
end

%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES

disp( 'Building NLSA model...' ); t = tic;
model = spczRainfall_nlsaModel( dataset, domain, period, inputVars ) 
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

