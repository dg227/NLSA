function y = lsum2_par( yL, idxE, nPar )
%LSUM2_PAR Lagged embedding sum along diagonal blocks using parallel for loops
%
% Modified 2020/07/11

if nargin == 2
    nPar = 0;
end

[ nSL1, nSL2 ] = size( yL );
nEL = max( idxE ) - 1;
nS1 = nSL1 - nEL;
nS2 = nSL2 - nEL;

y = zeros( nS1, nS2 );
parfor( iS = 1 : nS2, nPar )
    for iE = 1 : numel( idxE )
        y( :, iS ) = y( :, iS ) ...
                   + yL( idxE( iE ) : nS1 + idxE( iE ) - 1, ...
                         idxE( iE ) + iS - 1 ); 
    end
end
%parfor( iE = 1 : numel( idxE ), nPar )
%    y = y + yL( idxE( iE ) : nS1 + idxE( iE ) - 1, ...
%                idxE( iE ) : nS2 + idxE( iE ) - 1 );
%end

