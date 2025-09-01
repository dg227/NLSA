function path = getDensityPath( obj )
% GETDENSITYPATH  Get density path of an nlsaKernelDensity_fb object
%
% Modified 2018/07/06

path = fullfile( getPath( obj ), getDensitySubpath( obj ) );
