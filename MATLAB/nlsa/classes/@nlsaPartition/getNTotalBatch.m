function nB = getNTotalBatch( obj )
% GETNTOTALBATCH  Get the total number of batches in an array of 
% nlsaPartition objects
%
% Modified  2014/04/09

nB = getNBatch( obj );
nB = sum( nB( : ) );
