function obj = setDefaultTag( obj )
% SETDEFAULTTAG Set default tag of an nlsaKernelOperator object 
%
% Modified 2014/07/16

obj  = setTag( obj, getDefaultTag( obj ) );
