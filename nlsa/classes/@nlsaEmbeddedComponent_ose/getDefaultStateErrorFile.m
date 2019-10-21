function file = getDefaultStateErrorFile( obj )
% GETDEFAULTSTATEERRORFILE Get default state error files for an 
% nlsaEmbeddedComponent_ose object
%
% Modified 2014/05/20


file = getDefaultFile( getStateErrorFilelist( obj ), ...
                       getPartition( obj ), 'dataEX' );
