function obj = setOperatorFile( obj, file )
% SETOPERATORFILE  Set operator file of nlsaDiffusionOperator_gl object
%
% Modified 2014/04/08

if ~ischar( file )
    error( 'File must be a character string' )
end
obj.fileP = file;
