function obj = setVelocityErrorSubpath( obj, pathEXi )
% SETVELOCITYERRORSUBPATH  Set phase space velocity error subpath for an
% nlsaEmbeddedComponent_xi object
%
% Modified 2013/04/06

if ~isrowstr( pathEXi )
    error( 'pathXi property must be a character string' )
end
obj.pathEXi = pathEXi;
