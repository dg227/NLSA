function obj = setDataFile_before( obj, file )
% SETDATAFILE_BEFORE Set data filename for samples before the main time 
% interval of an nlsaEmbeddedComponent object
%
% Modified 2014/03/30

if ~isrowstr( file )
    error( 'Filename must be a character string' )
end
obj.fileB = file;

