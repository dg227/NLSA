function y = dmat( x1, x2 )
%DMAT distance matrix
%
%   Given two clouds of points in nD-dimensional space, represented by the 
%   arrays x1 and x2, respectively of size [ nD, nX1 ] and [ nD, nX2 ], 
%   y = dmat( x1, x2 ) returns the distance matrix y of size ( nX1. nX2 ) 
%   such that 
%
%   y( i, j ) = norm( x1( : , i ) - x2( :, j ) ) ^ 2.
%
%   The syntax y = dmat( x1 ) is equivalent to y = dmat( x1, x2 )
%    
%   Modified 04/03/2012

switch nargin
    
    case 1
    
        y = - 2 * x1' * x1;
        w = sum( x1 .^ 2, 1 );
        y = bsxfun( @plus, y, w );
        y = bsxfun( @plus, y, w' );
        y = abs( y + y' ) / 2; % Iron-out numerical wrinkles

    case 2
        
        y = - 2 * x1' * x2;
        w = sum( x1 .^ 2, 1 )';
        y = bsxfun( @plus, y, w );
        w = sum( x2 .^ 2, 1 );
        y = bsxfun( @plus, y, w );

end
