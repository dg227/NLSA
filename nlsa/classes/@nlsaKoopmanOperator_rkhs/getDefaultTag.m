function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaKoopmanOperator_rkhs objects
%
% Modified 2020/08/01

tag = getDefaultTag@nlsaKoopmanOperator( obj );

tag = [ tag sprintf( 'rkhs_%s_eps%1.3g', getRegularizationType( obj ), ...
                                         getRegularizationParameter( obj ) ) ];
