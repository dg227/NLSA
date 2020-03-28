function computePairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTEPAIRWISEDISTANCES Compute pairwise distances of nlsaModel
% 
% Modified 2020/03/28

if nargin == 1
    iProc = 1;
    nProc = 1;
end

embQ = getEmbComponentQ( obj );
embT = getEmbComponentT( obj );

pDistance = getPairwiseDistance( obj );

nB        = getNTotalBatch( pDistance );
pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iProc, nProc );

dDataQ    = nlsaLocalDistanceData( 'component', getEmbComponentQ( obj ) );

if ~isempty( embT )
    dDataT = nlsaLocalDistanceData( 'component', embT ); 
    
    computePairwiseDistances( pDistance, dDataQ, dDataT, ...
                              'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                              'logPath', getDistancePath( pDistance ), ...
                              'logFile', logFile, ...
                              varargin{ : } );
else
    computePairwiseDistances( pDistance, dDataQ,  ...
                              'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                              'logPath', getDistancePath( pDistance ), ...
                              'logFile', logFile, ...
                              varargin{ : } );
end
