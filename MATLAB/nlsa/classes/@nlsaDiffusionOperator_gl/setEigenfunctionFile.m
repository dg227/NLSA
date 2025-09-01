function obj = setEigenfunctionFile( obj, file )
% SETEIGENFUNCTIONFILE  Set eigenfunction file of an 
% nlsaDiffusionOperator_gl object
%
% Modified 2014/01/29

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.filePhi = file;
