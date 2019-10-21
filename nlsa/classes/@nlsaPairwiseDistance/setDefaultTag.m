function obj = setDefaultTag( obj )
% SETDEFAULTTAG  Set default tag of nlsaPairwiseDistance object
%
% Modified 2015/10/29

obj.dFunc = setTag( obj.dFunc, getDefaultTag( obj.dFunc ) );
obj = setTag( obj, getDefaultTag( obj ) );
