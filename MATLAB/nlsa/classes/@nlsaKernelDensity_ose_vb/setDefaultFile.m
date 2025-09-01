function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaKernelDensity_ose_vb object 
%
% Modified 2018/07/06

obj = setDefaultFile@nlsaKernelDensity_ose_fb( obj );
obj = setDistanceNormalizationFile( obj, getDefaultDistanceNormalizationFile( obj ) );
