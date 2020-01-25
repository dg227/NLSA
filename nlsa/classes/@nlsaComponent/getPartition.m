function partition = getPartition( obj )
% GETPARTITION  Get partition of nlsaComponent objects
%
% Modified 2020/01/25

% Return empty partition if empty object
if isempty( obj )
    partition = nlsaPartition.empty;
end 

% Return partitions of objects in the array
for iObj = numel( obj ) : -1 : 1
    partition( iObj ) = obj( iObj ).partition;
end
partition = reshape( partition, size( obj ) );  	
