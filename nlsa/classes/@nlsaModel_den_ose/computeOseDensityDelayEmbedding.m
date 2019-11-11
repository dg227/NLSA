function computeOseDensityDelayEmbedding( obj, iProcCD, nProcCD, iProcR, nProcR )
% COMPUTEOSEDENSITYDELAYEMBEDDING Lag-embed the OSE density in an 
% nlsaModel_den_ose object
%
% Modified 2019/11/10

if nargin == 1
    iProcCD = 1;
    nProcCD = 1;
    iProcR = 1;
    nProcR = 1;
end

cmp = getOseDensityKernel( obj );
emb = getOseEmbDensity( obj );
[ nCD, nR ] = size( emb );

pPartitionCD = nlsaPartition( 'nSample', nCD, 'nBatch', nProcCD );
pPartitionR = nlsaPartition( 'nSample', nR, 'nBatch', nProcR );
iCDLim = getBatchLimit( pPartitionCD, iProcCD );
iRLim = getBatchLimit( pPartitionR, iProcR );

for iR = iRLim( 1 ) : iRLim( 2 )
    for iC = iCDLim( 1 ) : iCDLim( 2 )
        computeData( emb( iC, iR ), cmp( iC ), iR )
    end
end
