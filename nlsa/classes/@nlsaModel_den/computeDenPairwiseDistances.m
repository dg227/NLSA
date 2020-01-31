function computeDenPairwiseDistances( obj, iProc, nProc, ...
                                           iDProc, nDProc, varargin )
% COMPUTEDENPAIRWISEDISTANCES Compute pairwise distances for the density data
% of nlsaModel_den objects
% 
% Modified 2020/01/29

if nargin == 1
    iProc = 1;
    nProc = 1;
end

if nargin <= 3
    iDProc = 1;
    nDProc = 1;
end

embComponentQ = getDenEmbComponentQ( obj ); 
embComponentT = getDenEmbComponentT( obj );
pDistance     = getDenPairwiseDistance( obj );
nB  = getNTotalBatch( pDistance( 1 ) );
nD  = numel( pDistance );
nCD = size( embComponentQ, 1 );

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
    dDataQ = nlsaLocalDistanceData( 'component', embComponentQ( iC, : ) );
    if ~isempty( embComponentT )
        dDataT = nlsaLocalDistanceData( 'component', embComponentT( iC, : ) );
    else
        dDataT = nlsaLocalDistanceData.empty;
    end
    computePairwiseDistances( pDistance( iD ), dDataQ, dDataT, ...
                              'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                              'logPath', getDistancePath( pDistance( iD ) ), ...
                              'logFile', logFile, ...
                              varargin{ : } );
end
