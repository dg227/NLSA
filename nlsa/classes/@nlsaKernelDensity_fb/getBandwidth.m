function epsilon = getBandwidth( obj )
% GETBANDWIDTH Get the kernel bandwidth of nlsaKernelDensity_fb objects
%
% Modified 2018/07/06

epsilon = getEpsilon( obj ) * computeOptimalBandwidth( obj );
