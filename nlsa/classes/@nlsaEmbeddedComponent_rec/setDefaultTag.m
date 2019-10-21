function obj = setDefaultTag( obj )
% SETDEFAULTTAG Set default tags of nlsaEmbeddedComponent_rec objects
%
% Modified 2014/08/04

obj = setDefaultTag@nlsaEmbeddedComponent_xi_e( obj );
obj = setBasisFunctionTag( obj, getDefaultBasisFunctionTag( obj ) );

