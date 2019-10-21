function obj = setVelocitySubpath( obj, pathXi )
% SETVELOCITYSUBPATH  Set phase space velocity subpath of nlsaEmbeddedComponent_xi object
%
% Modified 2013/04/04

if ~isrowstr( pathXi )
    error( 'pathXi property must be a character string' )
end
obj.pathXi = pathXi;
