%% INTEGRATE THE LORENZ 63 MODEL, AND OUTPUT POTENTIALLY PARTIAL OBSERVATIONS
%
% idxX specifies the component indices for the partial obs.
% Output is written in an array x of dimension [ nD nSProd ], where 
% nD = 3 in the case of full obs, nD = numel( idxX ) in the case of partial
% obs, and nSProd is the number of "production" samples
%
% Modified 2018-11-03

%% DATASET PARAMETERS
beta   = 8/3;           % L63 parameter beta
rho    = 28;            % L63 paramater rho
sigma  = 10;            % L63 parameter sigma 
nSProd = 64000;         % number of production samples
nSSpin = 128000;        % spinup samples
nEL    = 0;             % embedding window length (additional samples)
nXB    = 1;             % additional samples before production interval 
nXA    = 500;           % additional samples after production interval 
dt     = 0.01;          % sampling interval
x0     = [ 0 1 1.05 ];  % initial conditions
relTol = 1E-8;          % relative tolerance for ODE solver
ifCent = false;         % data centering
idxX   = { 1 [ 1 2 ] }; % state vector components for partial obs

%% SCRIPT EXECUTION OPTIONS
ifIntegrate = true;
ifPartialObs = true; 

%% NUMBER OF PRODUCTION SAMPLES AND OUTPUT PATH
nS = nSProd + nEL + nXB + nXA; 
strDir = [ 'beta'    num2str( beta, '%1.3g' ) ...
           '_rho'    num2str( rho, '%1.3g' ) ...
           '_sigma'  num2str( sigma, '%1.3g' ) ...
           '_dt'     num2str( dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', x0 ) ...
           '_nS'     int2str( nS ) ...
           '_nSSpin' int2str( nSSpin ) ...
           '_relTol' num2str( relTol, '%1.3g' ) ...
           '_ifCent' int2str( ifCent ) ];

pth = fullfile( './data', 'raw', strDir );
if ~isdir( pth )
    mkdir( pth )
end


%% INTEGRATE THE L63 SYSTEM
if ifIntegrate
    odeH = @( T, X ) l63( T, X, sigma, rho, beta );
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

    filenameOut = fullfile( pth, 'dataX.mat' );
    save( filenameOut, '-v7.3', 'x', 'mu' ) 
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

