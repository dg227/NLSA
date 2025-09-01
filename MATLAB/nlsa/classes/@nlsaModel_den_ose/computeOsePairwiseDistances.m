function computeOsePairwiseDistances( obj, iProc, nProc, varargin )
% COMPUTEOSEPAIRWISEDISTANCES Compute OSE pairwise distances of a 
% nlsaModel_den_ose object 
% 
% Modified 2020/01/30

if nargin == 1
    iProc = 1;
    nProc = 1;
end

dData  = nlsaLocalDistanceData_scl( 'component', getEmbComponentQ( obj ), ...
                                    'sclComponent', getEmbDensityQ( obj ) ); 
dOData = nlsaLocalDistanceData_scl( 'component', getOutEmbComponent( obj ), ...
                                    'sclComponent', getOseEmbDensity( obj ) ); 
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

