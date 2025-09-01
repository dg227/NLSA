function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of an nlsaEmbeddedComponent_d object
%
% Modified 2014/07/10

tag = strcat( getDefaultTag@nlsaEmbeddedComponent( obj ), 'diff' );
