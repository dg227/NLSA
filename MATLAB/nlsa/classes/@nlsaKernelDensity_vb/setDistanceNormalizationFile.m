function obj = setDistanceNormalizationFile( obj, file )
% SETDISTANCENORMALIZATIONFILE  Set distance normalization file of an nlsaKernelDensity_vb object
%
% Modified 2015/04/06

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileRho = file;
