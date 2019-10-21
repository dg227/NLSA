function computeIsrKernelNormalization( obj, varargin )
% COMPUTEISRKERNELNORMALIZATION Compute ISR kernel normalization of an 
% nlsaModel_err object  
% 
% Modified 2014/07/28


pDistance = getModPairwiseDistance( obj );
diffOp    = getIsrDiffusionOperator( obj );

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
    logFile  = sprintf( 'dataQ_%i-%i.log', iProc, nProc );
else
    logFile = 'dataQ_1_1.log';
end

computeNormalization( diffOp, pDistance, ...
                      'logPath', getNormalizationPath( diffOp ), ...
                      'logFile', logFile, ...
                      varargin{ : } );
