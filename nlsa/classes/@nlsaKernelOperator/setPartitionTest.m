function obj = setPartitionTest( obj, partition )
% SETPARTITIONTEST  Set test partition of an nlsaKernelOperator object
%
% Modified 2014/07/16

if ~isa( partition, 'nlsaPartition' )
    error( 'Second argument must be an nlsaPartition object' )
end

obj.partitionT = partition;
