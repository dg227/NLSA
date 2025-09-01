function nB = getNBatchTest( obj )
% GETNBATCHTEST  Get number of test batches in an nlsaKernelOperator object
%
% Modified  2014/07/16

nB = getNBatch( getPartitionTest( obj ) );
