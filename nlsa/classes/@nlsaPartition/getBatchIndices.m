function idx = getBatchIndices( obj, varargin )
% GETBATCHINDICES  Get indices for batches of nlsaPartition object
%
% Modified 2020/03/05

lim = getBatchLimit( obj, varargin{ : } );

nB = size( lim, 2 );
if nB == 1
    idx = lim( 1 ) : lim( 2 );
else
    idx = cell( 1, nB );
    for iB = 1 : nB
        idx{ iB } = lim( 1, iB ) : lim( 2, iB );
    end
end
