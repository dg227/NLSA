function pth = getDefaultDataSubpath( obj )
% GETDEFAULTDATASUBPATH Get default data subpath of nlsaEmbeddedComponent_e 
% object
%
% Modified 2014/04/06

pth = strcat( getDefaultDataSubpath@nlsaEmbeddedComponent( obj ), ...
              '_evector' ); 
