%% DYNAMICAL SYSTEM ON THE CIRCLE
%
% dtheta/dt = f * ( 1 - sqrt( 1 - aTheta ) * sin( theta ) )
%
% theta( 0 ) = 0
%
% The parameter values aPhi = aTheta = 1 correspond to linear flow on the circle
% 
% Modified 2019/10/20

%% DATASET PARAMETERS
f      = sqrt( 30 );   % frequency along theta coordinate
aTheta = 1;            % velocity variation parameter for theta
r      = 1;           % polar (theta) radius
nST    = 128;          % (approximate) samples per period
nT     = 64;           % number of periods
nTSpin = 0;            % spinup periods
nEL    = 0;            % extra samples for time-lagged embedding  
nXB    = 1;            % additional samples before main interval (for FD) 
nXA    = 1;            % additional samples after main interval (for FD)
idxX   = [ 1 : 2 ];    % coordinates to retain
obsMap = 'r2';         % observation map
ifCent = false;        % data centering

%% COMPUTE DATA
nS     = nST * nT  + nEL + nXB + nXA;  % total number of samples
nSSpin = nST * nTSpin; % spinup samples
nSTot  = nS + nSSpin; 
dt     = 2 * pi / nST; % sampling interval
t      = linspace( 0, ( nSTot - 1 ) * dt, nSTot ) / min( 1, f ); % timestamps

% Sample points on circle
if aTheta ~= 1
    theta = 2 * acot( ( sqrt( 1 - aTheta ) ) + sqrt( aTheta ) * cot( sqrt( aTheta ) * f * t / 2 ) ); 
else
    theta = f * t + pi / 2;
end
t = t( nSSpin + 1 : end );
theta = theta( nSSpin + 1 : end );

% Embedding in data space
% x is an array of size [ nD nS ] where nD is the dimension of data space
switch obsMap
    case 'r2' % Standard embedding of cicle in R^2
        x         = zeros( 2, nS );
        x( 1, : ) = r * cos( theta );
        x( 2, : ) = r * sin( theta );
end
x = x( idxX, : );

% Mean and data centering
mu = mean( x, 2 );
if ifCent
    x  = bsxfun( @minus, x, mu );
end

%% WRITE DATA

switch obsMap
    case 'r2' 
        strOut = [ 'r2' ...
                   '_r'     num2str( r, '%1.2f' ) ...
                   '_f',     num2str( f, '%1.2f' ) ...
                   '_aTheta' num2str( aTheta, '%1.2f' ) ...
                   '_dt'     num2str( dt ) ...
                   '_nS'     int2str( nS ) ...
                   '_nSSpin' int2str( nSSpin ) ...
                   '_idxX'   sprintf( '%i_', idxX ) ...  
                   'ifCent'  int2str( ifCent ) ];
end
pathOut = fullfile( './data', 'raw', strOut );
if ~isdir( pathOut )
    mkdir( pathOut )
end

filename = fullfile( pathOut, 'dataX.mat' );
save( filename, '-v7.3', 'x', 'mu', 'theta' ) 
