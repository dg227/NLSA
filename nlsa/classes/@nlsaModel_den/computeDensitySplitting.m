function computeDensitySplitting( obj, iProcCD, nProcCD )
% COMPUTEDENSITYSPLITTING Lag-embed the density in an nlsaModel_den object
%
% Modified 2020/01/28

if nargin == 1
    iProcCD = 1;
    nProcCD = 1;
end

cmp = getDensityKernel( obj );
cmpQ = getDensityKernelQ( obj );
nCD = size( cmp, 1 );

pPartitionCD = nlsaPartition( 'nSample', nCD, 'nBatch', nProcCD );
iCDLim = getBatchLimit( pPartitionCD, iProcCD );

for iC = iCDLim( 1 ) : iCDLim( 2 )
    splitData( cmpQ( iC, : ), cmp( iC ) )
end
