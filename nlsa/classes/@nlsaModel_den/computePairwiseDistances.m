function computePairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTEPAIRWISEDISTANCES Compute pairwise distances scaled by density for an 
% nlsaModel_den object
% 
% Modified 2015/10/28

if nargin == 1
    iProc = 1;
    nProc = 1;
end

emb = getEmbComponent( obj );
den = getEmbDensity( obj );
dData = nlsaLocalDistanceData_scl( 'component', emb, ...
                                   'sclComponent', den );
pDistance    = getPairwiseDistance( obj );

nB = getNTotalBatch( pDistance );
pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iBLim( 1 ), iBLim( 2 ) );

computePairwiseDistances( pDistance, dData, ...
                          'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                          'logPath', getDistancePath( pDistance ), ...
                          'logFile', logFile, ...
                          varargin{ : } );

