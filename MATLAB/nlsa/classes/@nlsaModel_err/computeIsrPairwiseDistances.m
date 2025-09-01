function computeIsrPairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTEISRPAIRWISEDISTANCES Compute pairwise distances for in-sample
% restriction (ISR) of an nlsaModel_err object
% 
% Modified 2014/07/28

embComponent    = getEmbComponent( obj ); 
outEmbComponent = getOutEmbComponent( obj );
pDistance       = getIsrPairwiseDistance( obj );

nB = getNTotalBatch( pDistance );
pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iProc, nProc );


computePairwiseDistances( pDistance, embComponent, outEmbComponent, ...
                          'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                          'logPath', getDistancePath( pDistance ), ...
                          'logFile', logFile, ...
                          varargin{ : } );

