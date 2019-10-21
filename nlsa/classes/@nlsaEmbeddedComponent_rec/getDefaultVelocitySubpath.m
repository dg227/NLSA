function pth = getDefaultVelocitySubpath( obj )
% GETDEFAULTVELOCITYSUBPATH Get default velocity subpath of an 
% nlsaEmbeddedComponent_rec object
%
% Modified 2014/07/06

pth = strcat( 'dataXi_r_', getDefaultTag( obj ) );
