function obj = setStateErrorSubpath( obj, pathEX )
% SETSTATEERRORSUBPATH  Set state error subpath for an nlsaEmbeddedComponent_ose
% object
%
% Modified 2013/05/20

if ~isrowstr( pathEX )
    error( 'pathX property must be a character string' )
end
obj.pathEXi = pathEX;
