function obj = setDataFile_after( obj, file )
% SETDATAFILE_AFTER Set data filename for samples after the main time 
% interval of an nlsaEmbeddedComponent object
%
% Modified 2014/03/30

if ~isrowstr( file )
    error( 'Filename must be a character string' )
end
obj.fileA = file;

