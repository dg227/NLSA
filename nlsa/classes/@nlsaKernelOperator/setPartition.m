function obj = setPartition( obj, partition )
% SETPARTITION Set partition of an nlsaKernelOperator object
%
% Modified 2014/07/16

if ~isa( partition, 'nlsaPartition' ) ...
        || ~isrow( partition )
    error( 'Partition must be a row vector of nlsaPartition objects' )
end

obj.partition = partition;

