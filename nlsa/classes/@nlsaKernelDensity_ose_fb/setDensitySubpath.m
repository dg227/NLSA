function obj = setDensitySubpath( obj, path )
% SETDENSITYSUBPATH  Set density subpath of an nlsaKernelDensity_ose_fb object
%
% Modified 2018/07/06

if ~isrowstr( path )
    error( 'Path must be a character string' )
end
obj.pathQ = path;
