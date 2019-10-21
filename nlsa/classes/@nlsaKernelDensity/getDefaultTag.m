function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaKernelDensity object
%
% Modified 2018/07/06

tag = sprintf( 'd%i_eps%1.2g', getDimension( obj ), getEpsilon( obj ) );

