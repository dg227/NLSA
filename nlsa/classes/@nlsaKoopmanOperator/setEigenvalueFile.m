function obj = setEigenvalueFile( obj, file )
% SETEIGENVALUEFILE  Set eigenvalue file of nlsaKoopmanOperator object
%
% Modified 2020/04/11

if ~ischar( file )
    error( 'File must be a character string' )
end
obj.fileEVal = file;
