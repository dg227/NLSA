function computeOutTrgDelayEmbedding( obj, iProcC, nProcC, iProcR, nProcR )
% COMPUTEOUTTRGDELAYEMBEDDING Lag-embed the out-of-sample target data in an 
% nlsaModel_ose object
%
% Modified 2020/08/02


if nargin == 1
    iProcC = 1;
    nProcC = 1;
    iProcR = 1;
    nProcR = 1;
end

cmp = getOutTrgComponent( obj );
emb = getOutTrgEmbComponent( obj );
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

