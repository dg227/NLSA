function obj = setPartitionTest( obj, partition )
% SETPARTITIONTEST  Set test partition of nlsaPairwiseDistance object
%
% Modified 2014/06/13

if ~isa( partition, 'nlsaPartition' ) ...
        || ~isrow( partition )
    error( 'Partition must be a row vector of nlsaPartition objects' )
end
obj.partitionT =  partition;
