function obj = setTemporalPatternFile( obj, file )
% SETTEMPORALPATTERNFILE  Set temporal pattern file of an nlsaLinearMap 
% object
%
% Modified 2015/10/19

if ~ischar( file )
    error( 'File must be a character string' )
end
obj.fileVT = file;
