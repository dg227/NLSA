function path = getDensityPath( obj )
% GETDENSITYPATH  Get density path of an nlsaKernelDensity_fb object
%
% Modified 2015/04/06

path = fullfile( getPath( obj ), getDensitySubpath( obj ) );
