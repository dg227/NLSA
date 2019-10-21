function tag = getTag( obj )
% GETTAG Get tags of an nlsaEmbeddedComponent object
%
% Modified 2014/07/29

tag = getTag@nlsaComponent( obj );
tag = { tag{ : } getEmbeddingTag( obj ) };
