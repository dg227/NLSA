function tag = getTag( obj )
% GETTAG Get tags of an nlsaEmbeddedComponent_ose object
%
% Modified 2015/12/14

tag = getTag@nlsaEmbeddedComponent( obj );
tag = { tag{ : } getOseTag( obj ) };
