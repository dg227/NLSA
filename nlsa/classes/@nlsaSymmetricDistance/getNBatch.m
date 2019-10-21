function nB = getNBatch( obj )
% GETNBATCH  Get number of batches in nlsaSymmetricDistance_batch object
%
% Modified  2014/05/01

nB = getNBatch( getPartition( obj ) );
