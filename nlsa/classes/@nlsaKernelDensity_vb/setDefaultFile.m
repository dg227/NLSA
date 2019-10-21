function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaKernelDensity_vb object 
%
% Modified 2015/12/16

obj = setDefaultFile@nlsaKernelDensity_fb( obj );
obj = setDistanceNormalizationFile( obj, getDefaultDistanceNormalizationFile( obj ) );
