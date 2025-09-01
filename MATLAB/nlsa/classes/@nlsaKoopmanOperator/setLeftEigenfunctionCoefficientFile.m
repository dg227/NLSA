function obj = setLeftEigenfunctionCoefficientFile( obj, file )
% SETLEFTEIGENFUNCTIONCOEFFICIENTFILE  Set left eigenfunction coefficient file 
% of an nlsaKoopmanOperator object
%
% Modified 2020/08/27

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileCoeffL = file;
