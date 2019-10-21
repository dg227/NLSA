function pth = getDefaultVelocitySubpath( obj )
% GETDEFAULTVELOCITYSUBPATH Get default phase space velocity subpath of an 
% nlsaEmbeddedComponent_xi object
%
% Modified 2014/08/04

pth = strcat( 'dataXi_', getDefaultVelocityTag( obj ) ); 
