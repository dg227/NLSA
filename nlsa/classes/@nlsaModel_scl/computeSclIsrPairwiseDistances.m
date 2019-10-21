function computeSclIsrPairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTESCLISRPAIRWISEDISTANCES Compute ISR scaled pairwise distances of an 
% nlsaModel_scl object
% 
% Modified 2014/07/28
`

embComponent    = getEmbComponent( obj ); 
isrErrComponent = getModErrComponent( obj );
oseEmbComponent = getOSEEmbComponent( obj );
oseErrComponent = getOSEErrComponent( obj );
pDistance       = getSclIsrPairwiseDistance( obj );

nB = getNTotalBatch( pDistance );
pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iProc, nProc );


computePairwiseDistances( pDistance, embComponent, isrErrComponent, ...
                                     oseEmbComponent, oseErrComponent, ...
                          'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                          'logPath', getDistancePath( pDistance ), ...
                          'logFile', logFile, ...
                          varargin{ : } );

