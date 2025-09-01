function computeOseDiffusionOperator( obj, iProc, nProc, varargin )
% COMPUTEOSEDIFFUSIONOPERATOR Compute out-of-sample extension (OSE) operator 
% of an nlsaModel_den_ose object
% 
% Modified 2018/07/04

if nargin == 1
    iProc = 1;
    nProc = 1;
end

oseDiffOp    = getOseDiffusionOperator( obj );
osePDistance = getOsePairwiseDistance( obj );
diffOp       = getDiffusionOperator( obj );

nB        = getNTotalBatch( getPartition( oseDiffOp ) );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataD_%i-%i.log', iProc, nProc );

computeOperator( oseDiffOp, osePDistance, diffOp, ...
                 'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                 'logPath', getOperatorPath( diffOp ), ...
                 'logFile', logFile, ...
                 varargin{ : } );
