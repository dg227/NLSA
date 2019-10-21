function obj = setPath( obj, path )
% SETPATH  Set path of an nlsaKernelOperator object
%
% Modified 2014/07/16

if ~isrowstr( path )
    error( 'Path property must be a character string' )
end
obj.path = path;
