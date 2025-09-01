function obj = setRightSingularVectorFile( obj, file )
% SETRIGHTSINGULARVECTORFILE  Set right singular vector file of an 
% nlsaCovarianceOperator_gl object
%
% Modified 2014/07/16

if ~ischar( file )
    error( 'File must be a character string' )
end
obj.fileV = file;
