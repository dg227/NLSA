function computeSclOutDiffusionOperator( obj, iProc, nProc, varargin )
% COMPUTESCLOUTDIFFUSIONOPERATOR Compute scaled diffusion operator for the
% OS data of an nlsaModel_scl object
% 
% Modified 2014/07/28

diffOp    = getSclOutDiffusionOperator( obj );
sDistance = getSclOutSymmetricDistance( obj );


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
