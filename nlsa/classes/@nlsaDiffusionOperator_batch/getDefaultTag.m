function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaDiffusionOperator_batch object
%
% Modified 2016/01/25

if isempty( obj.epsilonT ) || obj.epsilonT == obj.epsilon
    epsTag = sprintf( 'eps%1.2g', getEpsilon( obj ) );
else
    epsTag = sprintf( 'eps%1.2g_epsT%1.2g', getEpsilon( obj ), ...
                                            getEpsilonTest( obj ) );
end

if isempty( obj.nNT ) || obj.nNT == obj.nN
    nNTag = sprintf( 'nN%i', getNNeighbors( obj ) );
else
    nNTag = sprintf( 'nN%i_nNT%i', getNNeighbors( obj ), ...
                                   getNNeighborsTest( obj ) );
end

tag = sprintf( 'alpha%1.2f_%s_%s_nPhi%i', getAlpha( obj ), ...
                                          epsTag, ...
                                          nNTag, ...
                                          getNEigenfunction( obj ) );

