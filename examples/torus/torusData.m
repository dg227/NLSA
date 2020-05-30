%% DYNAMICAL SYSTEM ON THE TWO-TORUS
%
% dphi/dt   =       1 + sqrt( 1 - aPhi ) * cos( phi ) 
% dtheta/dt = f * ( 1 - sqrt( 1 - aTheta ) * sin( theta ) )
%
% phi( 0 )   = 0
% theta( 0 ) = 0
%
% The parameter values aPhi = aTheta = 1 correspond to linear flow on the torus
% 
% Modified 2016/02/02

%% DATASET PARAMETERS
f      = sqrt( 30 );   % frequency along theta coordinate
aPhi   = 1;            % velocity variation parameter for phi
aTheta = 1;            % velocity variation parameter for theta
r1     = .5;           % azimuthal (phi) radius
r2     = .5;           % polar (theta) radius
nST    = 128;          % (approximate) samples per period
nT     = 5;           % number of periods
nTSpin = 0;            % spinup periods
nEL    = 5;            % extra samples for time-lagged embedding  
nXB    = 1;            % additional samples before main interval (for FD) 
nXA    = 1;            % additional samples after main interval (for FD)
idxX   = [ 1 : 4 ];    % coordinates to retain
obsMap = 'r4';         % observation map
idxP   = 3;            % coordinates to apply deformation
p      = 0;            % deformation parameter 
ifCent = false;        % data centering

%% COMPUTE DATA
nS     = nST * nT  + nEL + nXB + nXA;  % total number of samples
nSSpin = nST * nTSpin; % spinup samples
nSTot  = nS + nSSpin; 
dt     = 2 * pi / nST; % sampling interval
t      = linspace( 0, ( nSTot - 1 ) * dt, nSTot ) / min( 1, f ); % timestamps

% Sample points on torus
% Azimuthal angle
if aPhi ~= 1
    phi   = 2 * atan( ( 1 + sqrt( 1 - aPhi ) ) * tan( sqrt( aPhi ) * t / 2 ) ...
                      ./ sqrt( aPhi ) );
else
    phi = t;
end
% Polar angle
if aTheta ~= 1
    theta = 2 * acot( ( sqrt( 1 - aTheta ) ) + sqrt( aTheta ) * cot( sqrt( aTheta ) * f * t / 2 ) ); 
else
    theta = f * t + pi / 2;
end
t = t( nSSpin + 1 : end );
phi = phi( nSSpin + 1 : end );
theta = theta( nSSpin + 1 : end );

% Embedding in data space
% x is an array of size [ nD nS ] where nD is the dimension of data space
switch obsMap
    case 'r3' % Standard embedding of 2-torus in R^3
        x            = zeros( 3, nS );
        x( 1, : )    = ( 1 + r1 * cos( theta ) ) .* cos( phi );
        x( 2, : )    = ( 1 + r1 * cos( theta ) ) .* sin( phi );
        x( 3, : )    = r2 * sin( theta );

        % Deformation
        if p > 0
            z            = exp( p * ( 1 + r1 - x( 1, : ) ) ...
                           .* ( 1 + r2 + x( 3, : ) ) );  
            x( idxP, : ) = bsxfun( @times, x( idxP, : ), ...
                                           z .^ ( 1 / numel( idxP ) ) );
        end

    case 'r4' % Flat embedding in R^4
        x = [ cos( phi ) 
              sin( phi ) 
              cos( theta ) 
              sin( theta ) ];
end
x = x( idxX, : );

% Mean and data centering
mu = mean( x, 2 );
if ifCent
    x  = bsxfun( @minus, x, mu );
end

%% WRITE DATA

switch obsMap
    case 'r3' % embedding in R^3  
        strOut = [ 'r3' ...
                   '_r1'     num2str( r1, '%1.2f' ) ...
                   '_r2'     num2str( r2, '%1.2f' ) ...
                   '_f',     num2str( f, '%1.2f' ) ...
                   '_aPhi'   num2str( aPhi, '%1.2f' ) ...
                   '_aTheta' num2str( aTheta, '%1.2f' ) ...
                   '_p'      num2str( p ) ...
                   '_idxP'  sprintf( '%i_', idxP ) ...
                   'dt'     num2str( dt ) ...
                   '_nS'     int2str( nS ) ...
                   '_nSSpin' int2str( nSSpin ) ...
                   '_idxX'   sprintf( '%i_', idxX ) ...  
                   'ifCent'  int2str( ifCent ) ];

    case 'r4' % flat embedding in R^4
        strOut = [ 'r4' ...
                   '_f',     num2str( f, '%1.2f' ) ...
                   '_aPhi'   num2str( aPhi, '%1.2f' ) ...
                   '_aTheta' num2str( aTheta, '%1.2f' ) ...
                   'dt'      num2str( dt ) ...
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
save( filename, '-v7.3', 'x', 'mu', 'theta', 'phi' ) 
