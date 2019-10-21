function obj = setDefaultTag( obj )
% SETDEFAULTTAG Set default tags of nlsaEmbeddedComponent_xi objects
%
% Modified 2014/08/04

obj = setDefaultTag@nlsaEmbeddedComponent( obj );
obj = setVelocityTag( obj, getDefaultVelocityTag( obj ) );

