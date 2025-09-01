function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaDiffusionOperator objects
%
% Modified 2021/02/27

if isempty( obj.epsilonT ) || obj.epsilonT == obj.epsilon
    epsTag = sprintf( 'eps%1.2g', getEpsilon( obj ) );
else
    epsTag = sprintf( 'eps%1.2g_epsT%1.2g', getEpsilon( obj ), ...
                                            getEpsilonTest( obj ) );
end

if obj.beta == 1
    betaTag = [];
else
    betaTag = sprintf( 'beta%1.2g_', getBeta( obj ) );
end

tag = sprintf( 'alpha%1.2f_%s%s_nPhi%i', getAlpha( obj ), ...
                                         betaTag, ...
                                         epsTag, ...
                                         getNEigenfunction( obj ) );
