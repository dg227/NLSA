function y = ldmat_scl( idxE, x1, s1, x2, s2 )
%LDMAT lagged distance matrix with scaling
%
% Modified 10/29/2015

r = zeros( 1, idxE( end ) );
r( idxE ) = 1;
r = diag( r );

switch nargin
    case 3
        y = dmat( x1 );
        y = bsxfun( @times, s1', y );
        y = bsxfun( @times, y, s1 );
    
    case 5
        y = dmat( x1, x2 );
        y = bsxfun( @times, s1', y );
        y = bsxfun( @times, y, s2 );
end

y = conv2( y, r, 'valid' );
