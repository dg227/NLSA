function computeSclOutPairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTESCLOUTPAIRWISEDISTANCES Compute scaled pairwise distances for the OS 
% data of an nlsaModel_scl object
% 
% Modified 2014/07/28

if nargin == 1
    iProc = 1;
    nProc = 1;
end

outEmbComponent = getOutEmbComponent( obj );
oseErrComponent = getOseErrComponent( obj );
pDistance       = getSclOutPairwiseDistance( obj );

nB = getNTotalBatch( pDistance );
pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iProc, nProc );


computePairwiseDistances( pDistance, outEmbComponent, oseErrComponent, ...
                          'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                          'logPath', getDistancePath( pDistance ), ...
                          'logFile', logFile, ...
                          varargin{ : } );

