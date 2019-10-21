function obj = setSingularValueFile( obj, file )
% SETSINGULARVALUEFILE  Set singular value file of an 
% nlsaCovarianceOperator object
%
% Modified 2014/07/16

if ~ischar( file )
    error( 'File must be a character string' )
end
obj.fileS = file;
