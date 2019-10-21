function partition = getPartitionTest( obj, iR )
% GETPARTITIONTEST  Get test partition of an nlsaKernelOperator object
%
% Modified 2016/01/25

if ~isempty( obj.partitionT )
    switch nargin
        case 1
            partition = obj.partitionT;
        case 2
            partition = obj.partitionT( iR );
    end
else 
    switch nargin
        case 1
            partition = getPartition( obj );
        case 2
            partition = getPartition( obj, iR );
    end
end
