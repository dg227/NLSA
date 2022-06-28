function dz = l96Multiscale( t, z, F, hX, hY, epsilon, nX, nY )
% Evaluates the right hand side of the 2-level Lorenz 96 system
%
% Modified 2021/07/16

z = z'; % make z a row vector

x = z( 1 : nX );       % slow variables 
y = z( nX + 1 : end ); % fast variables

y = reshape( y, [ nY nX ] );

% Vector field components for slow variables
dx = - circshift( x, -1 ) .* ( circshift( x, -2 ) - circshift( x, 1 ) ) ...
   - x + F + hX * mean( y, 1 );

% Vector field components for fast variables
dy = - circshift( y, 1 ) .* ( circshift( y, 2 ) - circshift( y, -1 ) ) ...
   - y + hY * x;
dy = dy / epsilon;

dy = reshape( dy, [ 1 nY * nX ] );

% Assemble into full vector field
dz = [ dx dy ];
dz = dz'; %  return a column vector

