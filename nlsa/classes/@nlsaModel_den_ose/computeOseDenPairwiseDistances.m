function computeOseDenPairwiseDistances( obj, iProc, nProc, ...
                                         iDProc, nDProc, varargin )
% COMPUTEOSEDENPAIRWISEDISTANCES Compute OSE pairwise distances for the 
% density data of nlsaModel_den_ose objects
% 
% Modified 2018/07/07

if nargin == 1
    iProc = 1;
    nProc = 1;
end

if nargin <= 3
    iDProc = 1;
    nDProc = 1;
end

outDenEmbComponent = getOutDenEmbComponent( obj ); 
denEmbComponent = getDenEmbComponent( obj );
pDistance    = getOseDenPairwiseDistance( obj );
nB = getNTotalBatch( pDistance( 1 ) );
nD = numel( pDistance );
nCD = size( denEmbComponent, 1 );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );

dPartition = nlsaPartition( 'nSample', nD, ...
                            'nBatch', nDProc );

iBLim = getBatchLimit( pPartition, iProc );
iDLim = getBatchLimit( dPartition, iDProc );

logFile = sprintf( 'dataY_%i-%i.log', iBLim( 1 ), iBLim( 2 ) );

for iD = iDLim( 1 ) : iDLim( 2 )
    if nD == 1 
        iC = 1 : nCD;
    else
        iC = iD;
    end
    dData = nlsaLocalDistanceData( 'component', denEmbComponent( iC, : ) );
    dOData = nlsaLocalDistanceData( 'component', outDenEmbComponent( iC, : ) );
    computePairwiseDistances( pDistance( iD ), dOData, dData, ...
                              'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                              'logPath', getDistancePath( pDistance( iD ) ), ...
                              'logFile', logFile, ...
                              varargin{ : } );
end
