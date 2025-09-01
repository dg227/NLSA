function obj = setDefaultTag( obj )
% SETDEFAULTTAG Set default tags of nlsaComponent_rec objects
%
% Modified 2015/08/31

obj = setBasisFunctionTag( obj, getDefaultBasisFunctionTag( obj ) );

