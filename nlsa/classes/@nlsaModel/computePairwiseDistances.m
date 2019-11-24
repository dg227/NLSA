function computePairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTEPAIRWISEDISTANCES Compute pairwise distances of nlsaModel
% 
% Modified 2019/11/24

if nargin == 1
    iProc = 1;
    nProc = 1;
end

dDataQ    = nlsaLocalDistanceData( 'component', getEmbComponentQ( obj ) );
dDataT    = nlsaLocalDistanceData( 'component', getEmbComponentT( obj ) ); 
pDistance = getPairwiseDistance( obj );
nB        = getNTotalBatch( pDistance );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iProc, nProc );


computePairwiseDistances( pDistance, dDataQ, dDataT, ...
                          'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                          'logPath', getDistancePath( pDistance ), ...
                          'logFile', logFile, ...
                          varargin{ : } );
