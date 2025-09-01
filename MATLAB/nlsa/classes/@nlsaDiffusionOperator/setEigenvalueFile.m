function obj = setEigenvalueFile( obj, file )
% SETEIGENVALUEFILE  Set eigenvalue file of nlsaDiffusionOperator object
%
% Modified 2014/04/08

if ~ischar( file )
    error( 'File must be a character string' )
end
obj.fileLambda = file;
