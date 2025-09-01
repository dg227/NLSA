function computeDiffusionOperator( obj, iProc, nProc, varargin )
% COMPUTEDIFFUSIONOPERATOR Compute diffusion operator of nlsaModel
% 
% Modified 2014/04/10

diffOp    = getDiffusionOperator( obj );
sDistance = getSymmetricDistance( obj );


if nargin > 1 && ~isempty( iProc )
    nB         = getNTotalEmbBatch( obj );
    pPartition = nlsaPartition( 'nSample', nB, ...
                                'nBatch',  nProc );
    iBLim      = getBatchLimit( pPartition, iProc );
    varargin = [ varargin { 'batch' iBLim( 1 ) : iBLim( 2 ) } ];
    logFile  = sprintf( 'dataP_%i-%i.log', iProc, nProc );
else
    logFile = 'dataP.log';
end

computeOperator( diffOp, sDistance, ...
                 'logPath', getOperatorPath( diffOp ), ...
                 'logFile', logFile, ...
                 varargin{ : } );
