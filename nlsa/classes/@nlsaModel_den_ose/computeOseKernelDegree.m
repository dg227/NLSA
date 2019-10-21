function computeOseKernelDegree( obj, varargin )
% COMPUTEOSEKERNELDEGREE Compute OSE kernel degree of nlsaModel_ose objects  
% 
% Modified 2018/07/04

if nargin == 1
    iProc = 1;
    nProc = 1;
end


pDistance = getOsePairwiseDistance( obj );
diffOp    = getDiffusionOperator( obj );
oseDiffOp = getOseDiffusionOperator( obj );

nB        = getNTotalBatch( getPartition( oseDiffOp ) );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataD_%i-%i.log', iProc, nProc );

computeDegree( oseDiffOp, pDistance, diffOp, ...
               'batch',   iBLim( 1 ) : iBLim( 2 ), ...
               'logPath', getDegreePath( oseDiffOp ), ...
               'logFile', logFile, ...
                varargin{ : } );
