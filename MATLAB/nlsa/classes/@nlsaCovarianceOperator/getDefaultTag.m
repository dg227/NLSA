function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of an nlsaCovarianceOperator object
%
% Modified 2016/02/02

tag = sprintf( 'cov_nL%i', getNEigenfunction( obj ) );
