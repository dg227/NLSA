function path = getDistancePath( obj )
% GETDISTANCEPATH  Get distance path of nlsaPairwiseDistance object
%
% Modified 2014/02/10

path = fullfile( getPath( obj ), getDistanceSubpath( obj ) );
