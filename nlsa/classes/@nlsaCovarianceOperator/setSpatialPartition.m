function obj = setSpatialPartition( obj, partition )
% SETDIMENSION  Set spatial partition of nlsaCovarianceOperator objects
%
% Modified 2014/08/07

if ~isa( partition, 'nlsaPartition' ) || ~isscalar( partition )
    error( 'Spatial partition must be specified as a scalar nlsaPartition object' )
end

obj.partitionD = partition;

if ~isCompatible( getLeftSingularVectorFilelist( obj ), partition )
    fList = nlsaFilelist( partition );
    obj = setLeftSingularVectorFile( obj, fList );
end

