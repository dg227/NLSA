function obj = setPath( obj, path )
% SETPATH  Set path of an nlsaKernelDensity object
%
% Modified 2015/04/06

if ~isrowstr( path )
    error( 'Path property must be a character string' )
end
obj.path = path;
