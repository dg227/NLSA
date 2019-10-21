function obj = setLinearMapFile( obj, file )
% SETLINEARMAPFILE  Set linear map file of an nlsaCovarianceOperator_gl object
%
% Modified 2014/07/16

if ~ischar( file )
    error( 'File must be a character string' )
end
obj.fileA = file;
