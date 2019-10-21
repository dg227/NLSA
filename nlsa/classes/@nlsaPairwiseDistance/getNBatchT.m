function nB = getNBatchT( obj )
% GETNBATCH  Get number of batches in test data in nlsaPairwiseDistance object
%
% Modified  2012/12/15

nB = getNBatch( obj.partitionT );