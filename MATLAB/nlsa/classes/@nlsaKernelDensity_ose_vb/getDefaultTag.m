function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaKernelDensity_ose_vb object
%
% Modified 2018/07/06

tag = sprintf( '_kNN%i', getKNN( obj ) );

tag = strcat( getDefaultTag@nlsaKernelDensity_ose_fb( obj ), tag );
