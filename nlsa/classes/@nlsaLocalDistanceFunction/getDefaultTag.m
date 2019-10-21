function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaLocalDistanceFunction object
%
% Modified 2015/10/29


tag = getDefaultTag( getLocalDistance( obj ) );
