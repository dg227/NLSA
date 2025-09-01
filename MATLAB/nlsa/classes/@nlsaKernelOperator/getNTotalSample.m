function nS = getNTotalSample( obj )
% GETNTOTALSAMPLE  Get the total number of samples in an arry of
% nlsaKernelOperator objects
%
% Modified 2014/07/16

nS = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nS( iObj ) = getNTotalSample( getPartition( obj( iObj ) ) );
end
