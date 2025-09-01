function obj = setDensityFile( obj, file )
% SETDENSITYFILE  Set file of an nlsaKernelDensity_ose_fb object
%
% Modified 2018/07/05

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileQ = file;
