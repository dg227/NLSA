function nB = getNTotalBatch( obj )
% GETNTOTALBATCH  Get total number of batches in an nlsaPairwiseDistance object
%
% Modified  2014/04/15

nB = getNTotalBatch( getPartition( obj ) );
