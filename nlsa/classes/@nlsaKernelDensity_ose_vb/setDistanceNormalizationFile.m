function obj = setDistanceNormalizationFile( obj, file )
% SETDISTANCENORMALIZATIONFILE  Set distance normalization file of an nlsaKernelDensity_ose_vb object
%
% Modified 2018/07/05

if ~isrowstr( file )
    error( 'File must be a character string' )
end
obj.fileRho = file;
