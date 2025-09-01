function pth = getDefaultDataSubpath( obj )
% GETDEFAULTDATASUBPATH Get default data subpath of an 
% nlsaEmbeddedComponent_overlap object
%
% Modified 2014/05/15

          
pth = strcat( getDefaultDataSubpath@nlsaEmbeddedComponent( obj ), ...
              '_overlap' ); 
