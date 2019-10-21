function computeOseKernelNormalization( obj, iProc, nProc, varargin )
% COMPUTEOSEKERNELNORMALIZATION Compute kernel normalization of 
% nlsaModel_den_ose objects  
% 
% Modified 2018/07/04

if nargin == 1
    iProc = 1;
    nProc = 1;
end


osePDistance = getOsePairwiseDistance( obj );
diffOp       = getDiffusionOperator( obj );
oseDiffOp    = getOseDiffusionOperator( obj );

nB        = getNTotalBatch( getPartition( oseDiffOp ) );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataQ_%i-%i.log', iProc, nProc );

computeNormalization( oseDiffOp, osePDistance, diffOp, ...
                      'batch',   iBLim( 1 ) : iBLim( 2 ), ...
                      'logPath', getNormalizationPath( oseDiffOp ), ...
                      'logFile', logFile, ...
                      varargin{ : } );

