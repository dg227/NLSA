function nS = getNTotalSample( obj )
% GETNTOTALSAMPLE  Get the total number of samples in nlsaSymmetricDistance 
% objects
%
% Modified 2014/04/03

nS = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nS( iObj ) = getNTotalSample( getPartition( obj( iObj ) ) );
end
