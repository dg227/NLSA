function nS = getNTotalSampleTest( obj )
% GETNTOTALSAMPLETEST  Get the total number of test samples in an arry of
% nlsaKernelOperator objects
%
% Modified 2014/07/16

nS = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nS( iObj ) = getNTotalSample( getPartitionTest( obj( iObj ) ) );
end
