function obj = setDoubleSumFile( obj, file )
% SETDOUBLESUMFILE  Set double sum file of an nlsaDiffusionOperator_gl_mb object
%
% Modified 2015/05/08

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileDSum = file;
