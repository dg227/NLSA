function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of an nlsaProjectedComponent object
%
% Modified 2014/06/20

tag = sprintf( 'nL%i', getNBasisFunction( obj ) );

