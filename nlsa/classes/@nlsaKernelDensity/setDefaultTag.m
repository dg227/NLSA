function obj = setDefaultTag( obj )
% SETDEFAULTTAG Set default tag of an nlsaKernelDensity object 
%
% Modified 2015/04/06

obj  = setTag( obj, getDefaultTag( obj ) );
