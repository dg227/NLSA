function obj = setDefaultTag( obj )
% SETDEFAULTTAG  Set default tag of nlsaLocalDistanceFunction object
%
% Modified 2015/10/31

obj.lDist = setTag( obj.lDist, getDefaultTag( obj.lDist ) );
