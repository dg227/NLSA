function obj = setPath( obj, path )
% SETPATH  Set path of nlsaComponent object
%
% Modified 2012/12/20

if ~ischar( path)
    error( 'Path property must be a character string' )
end
obj.path = path;
