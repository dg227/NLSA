function nB = getNBatch( obj )
% GETNBATCH  Get number of batches in an array of nlsaComponent objects
%
% Modified  2020/03/18

nB = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nB( iObj ) = getNBatch( obj( iObj ).partition );
end
