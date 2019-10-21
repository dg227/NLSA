function xE = lsum( x, iSLim, argE1, argE2 )
%LSUM Time-lagged sum
%
%   Modified 2018/05/04

switch nargin
    case 3
        % argE1 contains the indices for embedding
        idxE = argE1;
    case 4
        % argE1 is the embedding window length
        % argE2 is the embedding window sampling interval
        idxE = 1 : argE2 : argE1;
end

% Quick return if idxE == 1
if idxE == 1    
    xE = x( :, iSLim( 1 ) : iSLim( 2 ) );
    return
end

nD  = size( x, 1 );
nS  = size( x, 2 );
nSE = iSLim( 2 ) - iSLim( 1 ) + 1;

idxE = idxE( end ) + 1 - idxE;
idxE = iSLim( 1 ) - idxE;
[ I, J ] = ndgrid( 1 : nD, 1 : nSE );
k = sub2ind( [ nD nS ], I( : ), J( : ) )';
kShift = idxE * nD;
if ~iscolumn( kShift )
    kShift = kShift';
end
k = bsxfun( @plus, k, kShift );
xE = sum( x( k ), 1 );
xE = reshape( xE, [ nD nSE ] );


