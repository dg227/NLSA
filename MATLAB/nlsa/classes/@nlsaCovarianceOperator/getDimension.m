function nD = getDimension( obj, iC )
% GETDIMENSION  Get dimension of an nlsaCovarianceOperator object
%
% Modified 2015/10/20

partition = getSpatialPartition( obj );

if nargin == 1
    iC = 1 : getNBatch( partition );
end

nD = getBatchSize( partition, iC );


