%% LORENZ 96 MODEL

%% DATASET PARAMETERS
n      = 4;     % number of nodes
F      = 4;      % forcing parameter
dt     = 0.01;  % sampling interval 
nSProd = 102;   % number of "production" samples
nSSpin = 16000;  % spinup samples
nEL    = 0;      % embedding window length (additional samples)
nXB    = 1;      % additional samples before production interval (for FD)
nXA    = 1;      % additional samples after production interval (for FD)
x0      = zeros( 1, n );
x0( 1 ) = 1;
relTol = 1E-8;

odeH = @( T, X ) l96( T, X, F );
nS = nSProd + nEL + nXB + nXA; 
t = ( 0 : nS + nSSpin - 1 ) * dt;
[ tOut, xEns ] = ode45( odeH, t, x0, odeset( 'relTol', relTol, ...
                                          'absTol', eps ) );
xEns = xEns';
t = tOut';

t = t( nSSpin + 1 : end ) - t( nSSpin + 1 );
xEns = xEns( :, nSSpin + 1 : end );

%% WRITE DATA
strSrc = [ 'F'       num2str( F, '%1.3g' ) ...
           '_dt'     num2str( dt, '%1.3g' ) ...
           '_x0'     sprintf( '_%1.3g', x0( 1  ) ) ...
           '_nS'     int2str( nS ) ...
           '_nSSpin' int2str( nSSpin ) ...
           '_relTol' num2str( relTol, '%1.3g' ) ];

pathSrc = fullfile( './data', 'raw', strSrc );
if ~isdir( pathSrc )
    mkdir( pathSrc )
end

for iR = 1 : n
    x = xEns( iR, : );
    filename = fullfile( pathSrc, sprintf( 'dataX_%i.mat', iR ) );
    save( filename, '-v7.3', 'x', 't' )
end 

