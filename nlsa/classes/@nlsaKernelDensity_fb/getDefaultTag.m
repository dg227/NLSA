function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaKernelDensity_fb object
%
% Modified 2015/04/08

l = getBandwidthExponentLimit( obj );
tag = sprintf( '_b%1.2g_lLim%1.2g-%1.2g_nL%i', getBandwidthBase( obj ), ...
                                               l( 1 ), l( 2 ), ...
                                               getNBandwidth( obj ) );

tag = strcat( getDefaultTag@nlsaKernelDensity( obj ), tag );
