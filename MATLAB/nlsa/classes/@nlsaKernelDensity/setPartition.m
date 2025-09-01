function obj = setPartition( obj, partition )
% SETPARTITION Set partition of an nlsaKernelDensity object
%
% Modified 2015/04/06

if ~isa( partition, 'nlsaPartition' ) ...
        || ~isrow( partition )
    error( 'Partition must be a row vector of nlsaPartition objects' )
end

obj.partition = partition;

