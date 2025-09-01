function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaKernelDensity_fb object 
%
% Modified 2015/04/07

obj  = setDensityFile( obj, getDefaultDensityFile( obj ) );
obj  = setDoubleSumFile( obj, getDefaultDoubleSumFile( obj ) );
