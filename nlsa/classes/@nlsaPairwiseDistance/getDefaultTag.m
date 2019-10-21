function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaPairwiseDistance object
%
% Modified 2015/10/29

tag = sprintf( '%s_nN%i', getDefaultTag( getLocalDistanceFunction( obj ) ), ...
                          getNNeighbors( obj ) ); 
