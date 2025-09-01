function partition = getPartitionTest( obj )
% GETPARTITIONTEST  Get test data partition of nlsaPairwiseDistance object
%
% Modified 2016/01/25

if ~isempty( obj.partitionT )
    partition = obj.partitionT;
else 
    partition = getPartition( obj );
end
