function path = getDistancePath( obj )
% GETDISTANCEPATH  Get distance path of nlsaSymmetricDistance object
%
% Modified 2014/04/03

path = fullfile( getPath( obj ), getDistanceSubpath( obj ) );
