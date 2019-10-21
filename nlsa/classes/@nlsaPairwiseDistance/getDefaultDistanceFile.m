function file = getDefaultDistanceFile( obj )
% GETDEFAULTFILE Get default files of an nlsaPairwiseDistance object
%
% Modified 2014/01/07

file = getDefaultFile( getDistanceFilelist( obj ), ...
                       getPartition( obj ), 'dataY' );
