function tag = getTag( obj )
% GETTAG Get tags of an nlsaEmbeddedComponent_rec object
%
% Modified 2014/08/04

tag = getTag@nlsaEmbeddedComponent_xi_e( obj );
tag = { tag{ : } getBasisFunctionTag( obj ) };

