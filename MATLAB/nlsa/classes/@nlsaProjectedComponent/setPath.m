function obj = setPath( obj, path )
% SETPATH  Set path of an nlsaProjectedComponent object
%
% Modified 2014/06/20

if ~ischar( path)
    error( 'Path property must be a character string' )
end
obj.path = path;
