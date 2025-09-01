function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaKernelDensity_ose_fb object 
%
% Modified 2018/07/05

obj  = setDensityFile( obj, getDefaultDensityFile( obj ) );
