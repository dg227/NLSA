function obj = setDistanceFile( obj, file )
% SETDISTANCEFILE  Set data file of nlsaSymmetricDistance_gl object
%
% Modified 2014/04/07

if ~isrowstr( file )
    error( 'File property must be a character string' )
end
obj.file = file;
