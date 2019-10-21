%% LORENZ 96 MODEL

%% DATASET PARAMETERS
n      = 41;     % number of nodes
F      = 4;      % forcing parameter
dt     = 0.01;   % sampling interval 
nSProd = 1024; % number of "production" samples
nSSpin = 16000; % spinup samples
nEL    = 0;      % embedding window length (additional samples)
nXB    = 1;      % additional samples before production interval (for FD)
nXA    = 1;      % additional samples after production interval (for FD)
x0      = zeros( 1, n );
x0( 1 ) = 1;
relTol = 1E-8;
ifCent = false;        % data centering

odeH = @( T, X ) l96( T, X, F );
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
strSrc = [ 'F'       num2str( F, '%1.3g' ) ...
           '_n'      int2str( n ) ... 
           '_dt'     num2str( dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', x0( 1  ) ) ...
           '_nS'     int2str( nS ) ...
           '_nSSpin' int2str( nSSpin ) ...
           '_relTol' num2str( relTol, '%1.3g' ) ...
           '_ifCent' int2str( ifCent ) ];

pathSrc = fullfile( './data', 'raw', strSrc );
if ~isdir( pathSrc )
    mkdir( pathSrc )
end

filename = fullfile( pathSrc, 'dataX.mat' );
save( filename, '-v7.3', 'x', 't', 'mu' ) 

