function computeTrgDelayEmbedding( obj, iC, iR, nPar )
% COMPUTETRGDELAYEMBEDDING Lag-embed the target data in an nlsaModel_base object
%
% Modified 2020/08/31

cmp = getTrgComponent( obj );
emb = getTrgEmbComponent( obj );
[ nCT, nR ] = size( emb );

if nargin < 4 
    nPar = 0;
end

if nargin < 3 || isempty( iR )
    iR = 1 : nR;
end
if nargin < 2 || isempty( iC )
    iC = 1 : nCT;
end

[ iC, iR ] = ndgrid( iC, iR );

parfor( i = 1 : numel( iC ), nPar )
    computeData( emb( iC( i ), iR( i ) ), cmp( iC( i ), iR( i ) ) )
end
