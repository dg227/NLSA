function obj = setProjectionSubpath( obj, pth )
% SETPROJECTIONSUBPATH  Set projection subdirectory of an nlsaProjectedComponent
% object
%
% Modified 2014/06/20

if ~ischar( pth )
    error( 'Projection subdirectory must be a character string' )
end
obj.pathA = pth;
