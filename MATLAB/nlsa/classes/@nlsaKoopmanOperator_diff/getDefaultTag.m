function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaKoopmanOperator_diff objects
%
% Modified 2020/08/28

tag = getDefaultTag@nlsaKoopmanOperator( obj );

tag = [ tag sprintf( '_diff_%s_eps%1.3g', getRegularizationType( obj ), ...
                                          getRegularizationParameter( obj ) ) ];
