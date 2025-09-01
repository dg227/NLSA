function obj = setPartitionTest( obj, partition )
% SETPARTITIONTEST  Set test partition of nlsaPairwiseDistance object
%
% Modified 2020/01/25

if ~ ( isa( partition, 'nlsaPartition' )  ...
    && ( isrow( partition ) || isempty( partition ) ) )
    error( 'Partition must be a row vector of nlsaPartition objects, or an empty nlsaPartition object' )
end
obj.partitionT =  partition;
