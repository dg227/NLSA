function epsilon = getBandwidth( obj )
% GETBANDWIDTH Get the kernel bandwidth of nlsaDiffusionOperator_gl_mb objects
%
% Modified 2015/05/08

epsilon = getEpsilon( obj ) * computeOptimalBandwidth( obj );
