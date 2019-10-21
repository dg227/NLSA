function dx = l96( t, x, F )
% Evaluates the right hand side of the Lorenz 96 system
%
% dx( i ) = ( x( i + 1 ) - x( i - 2 ) ) * x( i - 1 ) - x( i ) + F

dx =  F - x ...
   + ( circshift( x, -1 ) - circshift( x, 2 ) ) .* circshift( x, 1 );

