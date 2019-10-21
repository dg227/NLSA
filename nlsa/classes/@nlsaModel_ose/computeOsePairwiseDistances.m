function computeOsePairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTEOSEPAIRWISEDISTANCES Compute OSE pairwise distances of a 
% nlsaModel_ose object 
% 
% Modified 2015/12/02

if nargin == 1
    iProc = 1;
    nProc = 1;
end

dData  = nlsaLocalDistanceData( 'component', getEmbComponent( obj ) ); 
dOData = nlsaLocalDistanceData( 'component', getOutEmbComponent( obj ) ); 
pDistance = getOsePairwiseDistance( obj );

nB = getNTotalBatch( pDistance );
pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iProc, nProc );

computePairwiseDistances( pDistance, dOData, dData, ...
                          'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                          'logPath', getDistancePath( pDistance ), ...
                          'logFile', logFile, ...
                          varargin{ : } );

