function w = wdmat_h( x1, xi1, x2, xi2 )
%WDMAT_H Distance matrix between data x1 and x2 weighted by the
%   harmonic mean of xi1^2 and xi2^2. 
%
%   Modified 12/12/2013

switch nargin
    case 2       
        xi1 = 1 ./ xi1 .^ 2;
        w   = dmat( x1 ) .* bsxfun( @plus, xi1', xi1 ) / 2;
    case 4
        xi1 = 1 ./ xi1 .^ 2;
        xi2 = 1 ./ xi2 .^ 2;
        w   = dmat( x1, x2 ) .* bsxfun( @plus, xi1', xi2 ) / 2;
end

