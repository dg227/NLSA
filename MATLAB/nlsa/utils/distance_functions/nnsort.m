function [ distSrt, idxSrt ] = nnsort( distIn, idxIn, nSrt, nPar )

if nargin == 3
    nPar = 0;
end

nS = size( distIn, 1 );

distSrt = zeros( nS, nSrt );
idxSrt  = zeros( nS, nSrt, class( idxIn ) );

parfor( iS = 1 : nS, nPar )
    [ distSrt( iS, : ), k  ] = mink( distIn( iS, : ), nSrt );
    idxSrt( iS, : )          = idxIn( iS, k );
end
