function obj = setOperatorFile( obj, file )
% SETOPERATORFILE  Set operator file of nlsaKoopmanOperator object
%
% Modified 2020/04/11

if ~ischar( file )
    error( 'File must be a character string' )
end
obj.fileOp = file;
