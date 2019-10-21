function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaDiffusionOperator objects
%
% Modified 2016/01/25

if isempty( obj.epsilonT ) || obj.epsilonT == obj.epsilon
    epsTag = sprintf( 'eps%1.2g', getEpsilon( obj ) );
else
    epsTag = sprintf( 'eps%1.2g_epsT%1.2g', getEpsilon( obj ), ...
                                            getEpsilonTest( obj ) );
end

tag = sprintf( 'alpha%1.2f_%s_nPhi%i', getAlpha( obj ), ...
                                       epsTag, ...
                                       getNEigenfunction( obj ) );
