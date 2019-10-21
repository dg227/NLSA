function computeIsrDiffusionOperator( obj, varargin )
% COMPUTEISRDIFFUSIONOPERATOR Compute in-sample-restriction (ISR) diffusion
% diffusion operator of an nlsaModel_err object
% 
% Modified 2014/07/28


diffOp       = getIsrDiffusionOperator( obj );
isrPDistance = getIsrPairwiseDistance( obj );
pDistance    = getOutPairwiseDistance( obj );

if nargin > 1 
    iProc = varargin{ 1 };
    nProc = varargin{ 2 };
    varargin = varargin( 3 : end );
end
if nargin > 1 && ~isempty( iProc )
    nB         = getNTotalBatch( getPartition( diffOp ) );
    pPartition = nlsaPartition( 'nSample', nB, ...
                                'nBatch',  nProc );
    iBLim      = getBatchLimit( pPartition, iProc );
    varargin = [ varargin { 'batch' iBLim( 1 ) : iBLim( 2 ) } ];
    logFile  = sprintf( 'dataP_%i-%i.log', iProc, nProc );
else
    logFile = 'dataP_1_1.log';
end

computeOperator( diffOp, isrPDistance, pDistance, ...
                   'logPath', getOperatorPath( diffOp ), ...
                   'logFile', logFile, ...
                   varargin{ : } );

