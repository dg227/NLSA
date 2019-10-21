function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaLocalScaling_pwr object
%
% Modified 2015/10/23

p = getExponent( obj );

tag = sprintf( 'pwr_p%1.2g', p );
