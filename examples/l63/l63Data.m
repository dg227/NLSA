%% Script to integrat the Lorenz 63 model. The output is written in an array x
% of dimension [ 3 nSProd ], where nSProd is the number of "production" samples

%% DATASET PARAMETERS
beta   = 8/3;          % L63 parameter beta
rho    = 28;           % L63 paramater rho
sigma  = 10;           % L63 parameter sigma 
nSProd = 640;64000;        % number of production samples
nSSpin = 6400;        % spinup samples
nEL    = 0;            % embedding window length (additional samples)
nXB    = 1;            % additional samples before production interval (for FD)
nXA    = 1;            % additional samples after production interval (for FD)
dt     = 0.01;         % sampling interval
x0     = [ 0 1 1.05 ]; % initial conditions
relTol = 1E-8;         % relative tolerance for ODE solver
ifCent = false;        % data centering

%% INTEGRATE THE L63 SYSTEM
odeH = @( T, X ) l63( T, X, sigma, rho, beta );
nS = nSProd + nEL + nXB + nXA; 
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

%% WRITE DATA
strSrc = [ 'beta'    num2str( beta, '%1.3g' ) ...
           '_rho'    num2str( rho, '%1.3g' ) ...
           '_sigma'  num2str( sigma, '%1.3g' ) ...
           '_dt'     num2str( dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', x0 ) ...
           '_nS'     int2str( nS ) ...
           '_nSSpin' int2str( nSSpin ) ...
           '_relTol' num2str( relTol, '%1.3g' ) ...
           '_ifCent' int2str( ifCent ) ];

pathSrc = fullfile( './data', 'raw', strSrc );
if ~isdir( pathSrc )
    mkdir( pathSrc )
end

filename = fullfile( pathSrc, 'dataX.mat' );
save( filename, '-v7.3', 'x', 'mu' ) 

