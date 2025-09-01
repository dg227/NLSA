function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of an nlsaSymmetricDistance object
%
% Modified 2014/04/10

if ~isa( partition, 'nlsaPartition' ) 
    error( 'Partition must be an nlsaPartition object' )
end
obj.partition = partition;
