unction symmetrizeOutDistances( obj, iProc, nProc, varargin )
% SYMMETRIZEMODDISTANCES Compute symmetric distance matrix from pairwise 
% distances of the OS data of an nlsaModel_err object
% 
% Modified 2014/05/25


pDist = getOutPairwiseDistance( obj );
sDist = getOutSymmetricDistance( obj );


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
