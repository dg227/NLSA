function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of an nlsaSymmetricDistance_batch object
%
% Modified 2014/06/13

if ~isa( partition, 'nlsaPartition' ) 
    error( 'Partition must be an nlsaPartition object' )
end
obj.partition = partition;

if ~isCompatible( getDistanceFilelist( obj ), partition )
    obj = setDistanceFile( obj, nlsaFilelist( partition ) );
end
