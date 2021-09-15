function w = wldmat_h( idxE, x1, xi1, x2, xi2 )
%WLDMAT Weighted lagged distance matrix with harmonic mean
%
%   Modified 05/12/2014

switch nargin
    
    case 3
        xi1 = 1 ./ xi1 .^ 2;
        w = ldmat( idxE, x1 ) .* bsxfun( @plus, xi1', xi1 ) / 2;

    case 5
        xi1 = 1 ./ xi1 .^ 2;
        xi2 = 1 ./ xi2 .^ 2;
        w = ldmat( idxE, x1, x2 ) .* bsxfun( @plus, xi1', xi2 ) / 2;
end

