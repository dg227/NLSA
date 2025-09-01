function obj = setDefaultTag( obj )
% SETDEFAULTTAG Set default tags of nlsaEmbeddedComponent objects
%
% Modified 2014/07/29

obj = setEmbeddingTag( obj, getDefaultEmbeddingTag( obj ) );

