function symmetrizeDistances( obj, iProc, nProc, varargin )
% SYMMETRIZEDISTANCES Compute symmetric distance matrix from pairwise distances
% 
% Modified 2014/05/01


pDist = getPairwiseDistance( obj );
sDist = getSymmetricDistance( obj );


if nargin > 1 && ~isempty( iProc )
    nB         = getNTotalBatch( getPartition( sDist ) );
    pPartition = nlsaPartition( 'nSample', nB, ...
                                'nBatch',  nProc );
    iBLim    = getBatchLimit( pPartition, iProc );
    varargin = [ varargin { 'batch' iBLim( 1 ) : iBLim( 2 ) } ];
    logFile  = sprintf( 'dataYS_%i-%i.log', iProc, nProc );
else
    logFile = 'dataYS.log';
end

symmetrizeDistances( sDist, pDist, ...
                     'logPath', getDistancePath( sDist ), ...
                     'logFile', logFile, ...
                      varargin{ : } )
