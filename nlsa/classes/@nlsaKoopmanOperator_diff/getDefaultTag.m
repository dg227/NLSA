function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaKoopmanOperator objects
%
% Modified 2020/04/08


tag = sprintf( 'dt%1.3g_eps%1.3g_nPhi%i', getSamplingInterval( obj ), ...
                                          getEpsilon( obj ), ...
                                          getNEigenfunction( obj ) );
