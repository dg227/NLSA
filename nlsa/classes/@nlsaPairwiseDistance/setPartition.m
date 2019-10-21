function obj = setPartition( obj, partition, partitionT )
% SETPARTITION  Set partition of nlsaPairwiseDistance object
%
% Modified 2014/06/13

if ~isa( partition, 'nlsaPartition' ) ...
        || ~isrow( partition )
    error( 'Partition must be a row vector of nlsaPartition objects' )
end
obj.partition = partition;

if nargin == 3
    obj = setPartitionTest( obj, partitionT );
end

if ~isCompatible( getDistanceFilelist( obj ), partition )
    obj = setDistanceFile( obj, nlsaFilelist( partition ) );
end
