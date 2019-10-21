%% KURAMOTO-SIVASHINSKY 9MODEL

%% DATASET PARAMETERS
L      = 18;     % system size (18 - periodic, 22 - chaos)
N      = 64;     % number of Fourier modes (= number of spatial gridpoints)
aIC    = 0.6;    % initial condition for Fourier modes
nIC    = 4;      % number of Fourier modes to be initialized to aIC
dt     = 0.25;   % sampling interval 
nSProd = 1E2;    % number of "production" samples
nSSpin = 1E4;    % spinup samples
nEL    = 15;     % embedding window length (additional samples)
nXB    = 1;      % additional samples before production interval (for FD)
nXA    = 1;      % additional samples after production interval (for FD)


%% COMPUTE DATA

nS = nSProd + nEL + nXB + nXA;

% Initial conditions
a0 = zeros( N - 2, 1 );  
a0( 1 : nIC ) = aIC; 

[ t, at ] = ksfmstp( a0, L, dt, nS + nSSpin, 1 );
[ s, ut ] = ksfm2real( at, L );

t  = t( nSSpin + 2 : end ) - t( nSSpin + 2 );
ut = ut( :, nSSpin + 2 : end );

%% WRITE DATA
strSrc = [ 'L'       num2str( L, '%1.3g' ) ...
           '_N'      int2str( N ) ...
           '_N0'     int2str( nIC ) ...
           '_a0'     sprintf( '_%1.3g', aIC ) ...
           '_dt'     num2str( dt, '%1.3g' ) ...
           '_nS'     int2str( nS ) ...
           '_nSSpin' int2str( nSSpin ) ];

pathSrc = fullfile( './data', 'raw', strSrc );
if ~isdir( pathSrc )
    mkdir( pathSrc )
end

for iR = 1 : N + 1
    x = ut( iR, : );
    sR = s( iR );
    filename = fullfile( pathSrc, sprintf( 'dataX_%i.mat', iR ) );
    save( filename, '-v7.3', 'x', 't', 'sR' )
end 

