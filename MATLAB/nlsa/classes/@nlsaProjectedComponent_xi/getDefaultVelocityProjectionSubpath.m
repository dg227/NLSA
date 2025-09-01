function pth = getDefaultVelocityProjectionSubpath( obj )
% GETDEFAULTVELOCITYPROJECTIONSUBPATH Get default velocity projection subpath 
% of an nlsaProjectedComponent_xi object
%
% Modified 2014/06/24

pth = strcat( 'dataAXi_', getDefaultTag( obj ) );
