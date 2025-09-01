function computeDenDelayEmbedding( obj, iProcC, nProcC, iProcR, nProcR )
% COMPUTEDENDELAYEMBEDDING Lag-embed the density data in an nlsaModel_den object
%
% Modified 2014/12/15

if nargin == 1
    iProcC = 1;
    nProcC = 1;
    iProcR = 1;
    nProcR = 1;
end

cmp = getDenComponent( obj );
emb = getDenEmbComponent( obj );
[ nC, nR ] = size( emb );

pPartitionC = nlsaPartition( 'nSample', nC, 'nBatch', nProcC );
pPartitionR = nlsaPartition( 'nSample', nR, 'nBatch', nProcR );
iCLim = getBatchLimit( pPartitionC, iProcC );
iRLim = getBatchLimit( pPartitionR, iProcR );

for iR = iRLim( 1 ) : iRLim( 2 )
    for iC = iCLim( 1 ) : iCLim( 2 )
        computeData( emb( iC, iR ), cmp( iC, iR ) )
    end
end
