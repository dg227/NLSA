function obj = setDefaultTag( obj )
% SETDEFAULTTAG Set default tags of nlsaEmbeddedComponent_ose objects
%
% Modified 2015/12/14

obj = setDefaultTag@nlsaEmbeddedComponent_xi( obj );
obj = setOseTag( obj, getDefaultOseTag( obj ) );

