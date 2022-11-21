function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaLocalScaling_pwr object
%
% Modified 2022/11/06

p = getExponent(obj);

tag = ['pwr' sprintf('_p%1.2g', p)];
