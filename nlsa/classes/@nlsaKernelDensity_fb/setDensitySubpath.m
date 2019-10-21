function obj = setDensitySubpath( obj, path )
% SETDENSITYSUBPATH  Set density subpath of an nlsaKernelDensity_fb object
%
% Modified 2015/04/06

if ~isrowstr( path )
    error( 'Path must be a character string' )
end
obj.pathQ = path;
