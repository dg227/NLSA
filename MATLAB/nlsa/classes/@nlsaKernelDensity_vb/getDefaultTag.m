function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaKernelDensity_vb object
%
% Modified 2015/04/07

tag = sprintf( '_kNN%i', getKNN( obj ) );

tag = strcat( getDefaultTag@nlsaKernelDensity_fb( obj ), tag );
