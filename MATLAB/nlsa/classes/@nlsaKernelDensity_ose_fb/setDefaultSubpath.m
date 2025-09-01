function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subdirectories of an nlsaKernelDensity_ose_fb 
% object 
%
% Modified 2018/07/05

obj = setDensitySubpath( obj, getDefaultDensitySubpath( obj ) );
