function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of an nlsaProjectedComponent object
%
% Modified 2014/06/24

if ~isa( partition, 'nlsaPartition' ) ...
        || ~isrow( partition )
    error( 'Partition must be a row vector of nlsaPartition objects' )
end
obj.partition = partition;
