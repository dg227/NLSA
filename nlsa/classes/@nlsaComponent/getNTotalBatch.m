function nS = getNTotalBatch( obj )
% GETNTOTALBATCH  Get the total number of batches in an array of 
% nlsaComponent objects
%
% Modified  2020/04/04

nB = getNBatch( obj );
nB = sum( nB( : ) );
