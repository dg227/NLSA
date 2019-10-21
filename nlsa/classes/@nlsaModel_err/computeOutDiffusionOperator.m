function computeOutDiffusionOperator( obj, iProc, nProc, varargin )
% COMPUTEMODDIFFUSIONOPERATOR Compute diffusion operator for the model data
% of an nlsaModel_err object
% 
% Modified 2014/05/26

diffOp    = getOutDiffusionOperator( obj );
sDistance = getOutSymmetricDistance( obj );


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
