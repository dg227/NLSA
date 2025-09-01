function w = wldmat_g( idxE, x1, xi1, x2, xi2 )
%WLDMAT Weighted lagged distance matrix with geometric mean
%
%   Modified 05/12/2014

switch nargin
    
    case 3
        w = ldmat( idxE, x1 );
        w = bsxfun( @rdivide, w, xi1' );
        w = bsxfun( @rdivide, w, xi1 );

    case 5
        
        w = ldmat( idxE, x1, x2 );
        w = bsxfun( @rdivide, w, xi1' );
        w = bsxfun( @rdivide, w, xi2 );
end

