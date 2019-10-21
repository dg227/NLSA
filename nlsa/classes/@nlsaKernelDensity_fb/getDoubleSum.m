function dSum = getDoubleSum( obj )
% GETDOUBLESUM  Read double sum of an nlsaKernelDensity_fb object
%
% Modified 2015/04/08

load( fullfile( getDensityPath( obj ), getDoubleSumFile( obj ) ), ...
      'dSum' )
