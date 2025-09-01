function computeOutPairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTESCLPAIRWISEDISTANCES Compute pairwise distances for the OS data 
% in an nlsaModel_err object
% 
% Modified 2014/07/24

if nargin == 1
    iProc = 1;
    nProc = 1;
end

embComponent = getOutEmbComponent( obj );
pDistance    = getOutPairwiseDistance( obj );

nB = getNTotalBatch( pDistance );
pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iProc, nProc );


computePairwiseDistances( pDistance, embComponent, ...
                          'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                          'logPath', getDistancePath( pDistance ), ...
                          'logFile', logFile, ...
                          varargin{ : } );

