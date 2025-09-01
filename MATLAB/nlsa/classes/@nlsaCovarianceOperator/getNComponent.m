function nC = getNComponent( obj )
% GETNCOMPONENT  Get number of components of an nlsaCovarianceOperator object
%
% Modified 2015/08/25

nC = getNBatch( getSpatialPartition( obj ) );
