function nB = getNBatch( obj )
% GETNBATCH  Get number of batches in an array of nlsaPartition objects
%
% Modified  2012/12/12

nB = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nB( iObj ) = numel( obj( iObj ).idx );
end
