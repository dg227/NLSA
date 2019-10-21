function file = getDefaultVelocityErrorFile( obj )
% GETDEFAULTVELOCITYERRORFILE Get default phase space velocity error files 
% for an nlsaEmbeddedComponent_ose object
%
% Modified 2014/04/06

file = getDefaultFile( getVelocityErrorFilelist( obj ), ...
                       getPartition( obj ), 'dataEXi' );
