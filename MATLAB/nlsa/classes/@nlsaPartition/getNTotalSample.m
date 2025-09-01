function nS = getNTotalSample( obj )
% GETNTOTALSAMPLE  Get the total number of samples in an array of 
% nlsaPartition objects
%
% Modified  2014/03/29

nS = getNSample( obj );
nS = sum( nS( : ) );
