function nB = getNBatch( obj )
% GETNBATCH  Get number of batches in an nlsaKernelOperator object
%
% Modified  2014/07/16

nB = getNBatch( getPartition( obj ) );
