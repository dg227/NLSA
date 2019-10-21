function obj = setVelocityProjectionSubpath( obj, pth )
% SETVELOCITYPROJECTIONSUBPATH  Set velocity projection subdirectory of an 
% nlsaProjectedComponent_xi object
%
% Modified 2014/06/24

if ~ischar( pth )
    error( 'Velocity projection subdirectory must be a character string' )
end
obj.pathAXi = pth;
