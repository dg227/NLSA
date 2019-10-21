function epsilon = getBandwidths( obj )
% GETBANDWIDTHS Get the kernel bandwidths of an nlsaKernelDensity object
%
% Modified 2015/05/07

l  = getBandwidthExponentLimit( obj );
nL = getNBandwidth( obj );
b  = getBandwidthBase( obj );
epsilon = linspace( l( 1 ), l( 2 ), nL );  
epsilon = b .^ epsilon;
