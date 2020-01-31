function computePairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTEPAIRWISEDISTANCES Compute pairwise distances scaled by density for an 
% nlsaModel_den object
% 
% Modified 2020/01/29

if nargin == 1
    iProc = 1;
    nProc = 1;
end

embQ = getEmbComponentQ( obj );
denQ = getEmbDensityQ( obj );
embT = getEmbComponentT( obj );
denT = getEmbDensityT( obj );

pDistance    = getPairwiseDistance( obj );

nB = getNTotalBatch( pDistance );
pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataY_%i-%i.log', iBLim( 1 ), iBLim( 2 ) );

dDataQ = nlsaLocalDistanceData_scl( 'component', embQ, ...
                                    'sclComponent', denQ );
if ~isempty( embT )
    dDataT = nlsaLocalDistanceData_scl( 'component', embT, ...
                                        'sclComponent', denT );
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

