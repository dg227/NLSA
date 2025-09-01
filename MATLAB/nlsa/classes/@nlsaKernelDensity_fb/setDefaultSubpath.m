function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subdirectories of an nlsaKernelDensity_fb 
% object 
%
% Modified 2015/04/06

obj = setDensitySubpath( obj, getDefaultDensitySubpath( obj ) );
