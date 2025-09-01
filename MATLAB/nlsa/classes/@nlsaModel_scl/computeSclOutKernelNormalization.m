function computeSclOutKernelNormalization( obj, varargin )
% COMPUTESCLOUTKERNELNORMALIZATION Compute scaled kernel normalization of 
% the OS data of an nlsaModel_scl object  
% 
% Modified 2014/07/28

logFile = 'dataQ.log';

sDistance = getSclOutSymmetricDistance( obj );
diffOp    = getSclOutDiffusionOperator( obj );

if nargin > 1 && ~isempty( iProc )
    nB         = getNTotalEmbBatch( obj );
    pPartition = nlsaPartition( 'nSample', nB, ...
                                'nBatch',  nProc );
    iBLim      = getBatchLimit( pPartition, iProc );
    varargin = [ varargin { 'batch' iBLim( 1 ) : iBLim( 2 ) } ];
    logFile  = sprintf( 'dataQ_%i-%i.log', iProc, nProc );
else
    logFile = 'dataQ.log';
end

computeNormalization( diffOp, sDistance, ...
                      'logPath', getNormalizationPath( diffOp ), ...
                      'logFile', logFile, ...
                      varargin{ : } );
