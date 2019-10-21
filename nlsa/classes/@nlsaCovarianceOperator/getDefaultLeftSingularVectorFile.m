function file = getDefaultLeftSingularVectorFile( obj )
% GETDEFAULTLEFTSINGULARVECTORFILE Get default left singular vector file of 
% an nlsaCovarianceOperator object
%
% Modified 2014/08/08

file = getDefaultFile( getLeftSingularVectorFilelist( obj ), ...
                       getSpatialPartition( obj ), 'dataU' );
