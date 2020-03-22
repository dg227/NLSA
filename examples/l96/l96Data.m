%% LORENZ 96 MODEL

% Modified 2019-04-05   

%% DATASET PARAMETERS
n      = 41;     % number of nodes
F      = 4;      % forcing parameter
dt     = 0.01;   % sampling interval 
nSProd = 64000;   % number of "production" samples
nSSpin = 768000;  % spinup samples
nEL    = 0;      % embedding window length (additional samples)
nXB    = 1;      % additional samples before production interval (for FD)
nXA    = 1;      % additional samples after production interval (for FD)
x0      = zeros( 1, n );
x0( 1 ) = 1;
relTol = 1E-8;
ifCent = false;        % data centering
idxX   = { 1 [ 1 21 ] [ 1 11 21 31 ] [ 1 6 11 15 21 26 31 36 ] };    % state vector components for partial obs 

%% SCRIPT EXECUTION OPTIONS
ifIntegrate = true;
ifPartialObs = true; 

%% NUMBER OF PRODUCTION SAMPLES AND OUTPUT PATH
nS = nSProd + nEL + nXB + nXA; 
strSrc = [ 'F'       num2str( F, '%1.3g' ) ...
           '_n'      int2str( n ) ... 
           '_dt'     num2str( dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', x0( 1  ) ) ...
           '_nS'     int2str( nS ) ...
           '_nSSpin' int2str( nSSpin ) ...
           '_relTol' num2str( relTol, '%1.3g' ) ...
           '_ifCent' int2str( ifCent ) ];

pth = fullfile( './data', 'raw', strSrc );
if ~isdir( pth )
    mkdir( pth )
end

%% INTEGRATE THE L96 MODEL
if ifIntegrate
    odeH = @( T, X ) l96( T, X, F );
    t = ( 0 : nS + nSSpin - 1 ) * dt;
    [ tOut, x ] = ode45( odeH, t, x0, odeset( 'relTol', relTol, ...
                                          'absTol', eps ) );
    x = x';
    t = tOut';

    t = t( nSSpin + 1 : end ) - t( nSSpin + 1 );
    x = x( :, nSSpin + 1 : end );

    mu = mean( x, 2 );
    if ifCent
        x  = bsxfun( @minus, x, mu );
    end

    filename = fullfile( pth, 'dataX.mat' );
    save( filename, '-v7.3', 'x', 't', 'mu' ) 
end


%% CREATE DATASETS WITH PARTIAL OBS
if ifPartialObs
    filenameIn = fullfile( pth, 'dataX.mat' );
    Raw = load( filenameIn, 'x' ); 

    for iObs = 1 : numel( idxX )
        x = Raw.x( idxX{ iObs }, : );
        filenameOut = fullfile( pth, ...
                                strcat( 'dataX_idxX', sprintf( '_%i', idxX{ iObs } ), ...
                                '.mat' ) );  
        save( filenameOut, '-v7.3', 'x' )
    end
end

