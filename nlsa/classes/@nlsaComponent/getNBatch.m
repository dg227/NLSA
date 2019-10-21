function nB = getNBatch( obj )
% GETNBATCH  Get number of batches in an array of nlsaComponent objects
%
% Modified  2012/10/24

nB = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nB( iObj ) = getNBatch( obj.partition( iObj ) );
end
