function obj = setPath( obj, path )
% SETPATH  Set path of an nlsaSymmetricDistance object
%
% Modified 2014/04/03

if ~isrowstr( path )
    error( 'Path property must be a character string' )
end
obj.path = path;
