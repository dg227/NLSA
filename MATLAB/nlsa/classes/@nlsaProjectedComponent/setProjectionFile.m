function obj = setProjectionFile( obj, file )
% SETPROJECTIONFILE  Set projected data file of an nlsaProjected component
% object
%
% Modified 2014/06/20

if ~isrowstr( file )
    error( 'File property must be a character string' )
end
obj.fileA = file;
