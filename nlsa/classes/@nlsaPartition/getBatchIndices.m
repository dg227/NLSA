function idx = getBatchIndices( obj, varargin )
% GETBATCHINDICES  Get indices of batches of an nlsaPartition object.
%
% Modified 2020/03/05

lim = getBatchLimit( obj, varargin{ : } );

nB = size( lim, 1 );
if nB == 1
    idx = lim( 1 ) : lim( 2 );
else
    idx = cell( nB, 1 );
    for iB = 1 : nB
        idx{ iB } = lim( iB, 1 ) : lim( iB, 2 );
    end
end
