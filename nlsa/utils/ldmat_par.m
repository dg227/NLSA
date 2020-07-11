function y = ldmat_par( idxE, x1, x2, nPar )
%LDMAT_PAR lagged distance matrix with support for parallel for loops
%
% Modified 2020/07/11

if nargin < 4
    nPar = 0; 
end

if nargin == 2 || isempty( x2 )
    yL  = dmat( x1 );
else
    yL  = dmat( x1, x2 );
end

y = lsum2_par( yL, idxE, nPar );
