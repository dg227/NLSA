function computeSclIsrDiffusionOperatorOperator( obj, varargin )
% COMPUTESCLISRDIFFUSIONOPERATOR Compute scaled ISR diffusion operator 
% of an nlsaModel_scl object
% 
% Modified 2014/07/28


diffOp       = getSclIsrDiffusionOperator( obj );
prjPDistance = getSclIsrPairwiseDistance( obj );
pDistance    = getSclOutPairwiseDistance( obj );

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

computeOperator( diffOp, prjPDistance, pDistance, ...
                   'logPath', getOperatorPath( diffOp ), ...
                   'logFile', logFile, ...
                   varargin{ : } );

