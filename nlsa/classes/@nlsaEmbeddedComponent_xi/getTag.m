function tag = getTag( obj )
% GETTAG Get tags of an nlsaEmbeddedComponent_xi object
%
% Modified 2014/08/04

tag = getTag@nlsaEmbeddedComponent( obj );
tag = { tag{ : } getVelocityTag( obj ) };

