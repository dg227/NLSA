function pth = getDefaultVelocitySubpath( obj )
% GETDEFAULTVELOCITYSUBPATH Get default velocity subpath of an 
% nlsaEmbeddedComponent_xi_e object
%
% Modified 2014/05/15

pth = strcat( getDefaultVelocitySubpath@nlsaEmbeddedComponent_xi( obj ), ...
              '_evector' ); 
