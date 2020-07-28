% NLSA OF PANGAEA DATA
% 
% Commands to retrieve output from the NLSA model:
%
% phi   = getDiffusionEigenfunctions( model ); -- NLSA eigenfunctions
%
% Modified 2020/07/27

%% DATA SPECIFICATION 
sourceVar = 'temp';   % global mean temperature
embWindow = '100ka';  % 100,000 year embedding
kernel    = 'l2';     % L2 kernel      


%% SCRIPT EXECUTION OPTIONS
% Data extraction
ifDataSource = true;  % extract source precipitation data from NetCDF files  

% Eigenfunctions
ifNLSA    = true; % compute kernel (NLSA) eigenfunctions

% Output/plotting options
ifPrintFig        = true;      % print figures to file

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes


%% EXTRACT SOURCE DATA
if ifDataSource
    disp( sprintf( 'Reading source data %s...', sourceVar ) ); t = tic;
    pangaeaAnalysis_data( sourceVar ) 
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
experiment = { sourceVar [ embWindow 'Emb' ] ...
               [ kernel 'Kernel' ] };
experiment = strjoin_e( experiment, '_' );

disp( 'Building NLSA model...' ); t = tic;
model = pangaeaAnalysis_nlsaModel( experiment );
toc( t )

nSE          = getNTotalSample( model.embComponent );
nSB          = getNXB( model.embComponent );
nEL          = getEmbeddingWindow( model.embComponent ) - 1;
nShiftTakens = round( nEL / 2 );


switch experiment

case 'temp_100kaEmb_l2Kernel'

    Plt.idxPhi = [ 2 3 4 ];  % eigenfunctions to plot
    Plt.tLim   = [ 1 1000 ]; % time interval to plot

otherwise
    error( 'Invalid experiment.' )

end

% Figure directory
figDir = fullfile( pwd, 'figs', experiment );
if ifPrintFig && ~isdir( figDir )
    mkdir( figDir )
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


