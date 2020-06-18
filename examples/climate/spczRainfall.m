% NLSA/KOOPMAN ANALYSIS OF SPCZ RAINFALL
% 
% Commands to retrieve output from the NLSA model:
%
% phi   = getDiffusionEigenfunctions( model ); -- NLSA eigenfunctions
% z     = getKoopmanEigenfunctions( model );   -- Koopman eigenfunctions
% gamma = getKoopmanEigenvalues( model ) * 12 / (2*pi) -- Koopman eigenvalues  
% T     = getKoopmanEigenperiods( model ) / 12; -- Koopman eigenperiods
% uPhi  = getProjectedData( model ); -- Projected data onto NLSA eigenfunctons
% uZ    = getKoopmanProjectedData( model ); -- Proj. data onto Koopman eigenfunctions
%
% Koopman eigenfrequencies are the imaginary part of gamma, and are in units
% of 1/year.
%
% Koopman eigenperiods (T) are in units of year. 
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

% Eigenfunctions
ifNLSA    = true; % compute kernel (NLSA) eigenfunctions
ifKoopman = true; % compute Koopman eigenfunctions

% Koopman spectrum
ifKoopmanSpectrum = true;  % plot generator spectrum

% Output/plotting options
ifPrintFig        = true;      % print figures to file

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

switch experiment

case 'cmap_satellite_PacPrecip_4yrEmb_coneKernel'

    % Parameters for spectrum plots
    Spec.mark = { 1          ... % constant
                  [ 2 3 ]    ... % annual
                  [ 4 5 ]    ... % semiannual
                  [ 6 7 ]    ... % ENSO
                  [ 13 : 16 ] ... % ENSO combination
                 };
    Spec.legend = { 'mean' ... 
                    'annual' ...
                    'semiannual' ...
                    'ENSO' ... 
                    'ENSO combination' ... 
                  };
    Spec.xLim = [ -1.5 .1 ];
    Spec.yLim = [ -3 3 ]; 
    Spec.c = distinguishable_colors( numel( Spec.mark ) );

otherwise
    error( 'Invalid experiment.' )

end

% Figure directory
figDir = fullfile( pwd, 'figs', experiment );
if ~isdir( figDir )
    mkdir( figDir )
end



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

    disp( 'Projection of target data onto kernel eigenfunctions...' ); t = tic;
    computeProjection( model )
    toc( t )

end

%% COMPUTE EIGENFUNCTIONS OF KOOPMAN GENERATOR
if ifKoopman
    disp( 'Koopman eigenfunctions...' ); t = tic;
    computeKoopmanEigenfunctions( model )
    toc( t )

    disp( 'Projection of target data onto Koopman eigenfunctions...' ); t = tic;
    computeKoopmanProjection( model )
    toc( t )
end

%% PLOT OF GENERATOR SPECTRUM
if ifKoopmanSpectrum

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 6; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = 2.1;
    Fig.deltaY     = .48;
    Fig.deltaY2    = .2;
    Fig.gapX       = .20;
    Fig.gapY       = .5;
    Fig.gapT       = .25; 
    Fig.nTileX     = 1;
    Fig.nTileY     = 1;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );


    % Get generator eigenvalues in units of 1 / year
    gamma = getKoopmanEigenvalues( model ) * 12 / 2 / pi; 

    % Plot marked eigenvalues
    ifMarked = false( size( gamma ) );
    for iMark = 1 : numel( Spec.mark )
        ifMarked( Spec.mark{ iMark } ) = true;
        plot( real( gamma( Spec.mark{ iMark } ) ), ...
              imag( gamma( Spec.mark{ iMark } ) ), '.', 'markersize', 15, ...
              'color', Spec.c( iMark, : ) )
    end
    
    % Plot unmarked eigenvalues
    plot( real( gamma( ~ifMarked ) ), imag( gamma( ~ifMarked ) ), ...
          '.', 'markerSize', 10, 'color', [ .5 .5 .5 ] )

    grid on
    xlim( Spec.xLim )
    ylim( Spec.yLim )
    title( 'Generator spectrum' )
    ylabel( 'frequency (1/y)' )
    xlabel( 'decay rate (arbitrary units)' )
    axPos = get( gca, 'position' );
    hL = legend( Spec.legend, 'location', 'eastOutside' );
    set( gca, 'position', axPos )

    % Print figure
    if ifPrintFig
        figFile = fullfile( figDir, 'figGeneratorSpectrum.png'  );
        print( fig, figFile, '-dpng', '-r300' ) 
    end

end


