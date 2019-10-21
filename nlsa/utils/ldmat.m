function y = ldmat( idxE, x1, x2 )
%LDMAT lagged distance matrix
%
% Modified 2018/05/04

nS1 = size( x1, 2 ) - idxE( end ) + 1;
switch nargin
    case 2
        nS2 = nS1;
        yL  = dmat( x1 );
    case 3
        nS2 = size( x2, 2 ) - idxE( end ) + 1;
        yL  = dmat( x1, x2 );
end

y = lsum2( yL, idxE );
