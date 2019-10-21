function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of nlsaComponent object
%
% Modified 2014/04/11

if ~isa( partition, 'nlsaPartition' ) || ~isscalar( partition )
    error( 'Partition must be a sclar nlsaPartition object' )
end
obj.partition = partition;

if getNBatch( partition ) ~= getNFile( getDataFilelist( obj ) )
    obj = setDataFile( obj, nlsaFilelist( 'nFile', getNBatch( partition ) ) );
end
