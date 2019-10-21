function partition = getPartition( obj )
% GETPARTITION  Get partition of nlsaComponent objects
%
% Modified 2017/04/02

for iObj = numel( obj ) : -1 : 1
    partition( iObj ) = obj( iObj ).partition;
end
partition = reshape( partition, size( obj ) );  	
