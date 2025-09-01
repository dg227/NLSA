function obj = setDensityFile( obj, file )
% SETDENSITYFILE  Set file of an nlsaKernelDensity_fb object
%
% Modified 2015/04/06

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileQ = file;
