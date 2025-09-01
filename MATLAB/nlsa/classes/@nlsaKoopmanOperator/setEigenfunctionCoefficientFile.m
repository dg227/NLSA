function obj = setEigenfunctionCoefficientFile( obj, file )
% SETEIGENFUNCTIONCOEFFICIENTFILE  Set eigenfunction coefficient file of an 
% nlsaKoopmanOperator object
%
% Modified 2020/04/15

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileCoeff = file;
