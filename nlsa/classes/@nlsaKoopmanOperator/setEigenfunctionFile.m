function obj = setEigenfunctionFile( obj, file )
% SETEIGENFUNCTIONFILE  Set eigenfunction file of an nlsaKoopmanOperator 
% object
%
% Modified 2020/04/11

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileEFunc = file;
