function w = wdmat_g( x1, xi1, x2, xi2 )
%WDMAT_G Distance matrix between data x1 and x2 weighted by the
%   squared geometric mean of xi1 and xi2. 
%
%   Modified 12/12/2013

switch nargin
    
    case 2
    
        w = dmat( x1 );
        w = bsxfun( @rdivide, w, xi1' );
        w = bsxfun( @rdivide, w, xi1 );

    case 4
        
        w = dmat( x1, x2 );
        w = bsxfun( @rdivide, w, xi1' );
        w = bsxfun( @rdivide, w, xi2 );
end

