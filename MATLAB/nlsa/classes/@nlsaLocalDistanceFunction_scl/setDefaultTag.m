function obj = setDefaultTag( obj )
% SETDEFAULTTAG  Set default tag of an nlsaLocalDistanceFunction_scl object
%
% Modified 2015/10/31

obj = setDefaultTag@nlsaLocalDistanceFunction( obj );
obj.lScl = setTag( obj.lScl, getDefaultTag( obj.lscl ) );
