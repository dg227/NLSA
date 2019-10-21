function [ u, v, x, y, c ] = state2vel( s, lX, lY )
% Convert dynamical system state s to a 2D incompressible velocity field (u, v) 
% using the cross sweep/shear approach of Majda et al. c are the spatial 
% Fourier coefficients of s.   

n = size( s, 1 );
N = size( s, 2 );

x = ( 0 : n - 1 ) * lX / n;
y = ( 0 : n - 1 ) * lY / n;
X = meshgrid( x, y );
X = X( : );

c = fft( s, [], 1 );

u = real( bsxfun( @times, ones( n ^ 2, 1 ), c( 1, : ) ) );
u = reshape( u, [ n n N ] );

v = zeros( n^2, N );
for r = 2 : n
    %v = real( bsxfun( @times, exp( 2 * pi * i * r * X / lX ), c( r, : ) ) ) ...
    v = real( bsxfun( @times, exp( 2 * pi * i * ( r - 1 ) * X / lX ), c( r, : ) ) ) ...
      + v;
end
v = reshape( v, [ n n N ] );



 
