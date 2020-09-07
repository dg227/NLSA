function obj = setLeftEigenfunctionFile( obj, file )
% SETLEFTEIGENFUNCTIONFILE  Set left eigenfunction file of an 
%  nlsaKoopmanOperator object
%
% Modified 2020/08/27

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileEFuncL = file;
