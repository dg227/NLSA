function nS = getNTotalSample( obj )
% GETNTOTALSAMPLE  Get the total number of samples in an array of
% nlsaKernelDensity objects
%
% Modified 2015/04/06

nS = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nS( iObj ) = getNTotalSample( getPartition( obj( iObj ) ) );
end
