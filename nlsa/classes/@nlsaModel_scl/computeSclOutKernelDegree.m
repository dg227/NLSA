function computeSclOutKernelDegree( obj, varargin )
% COMPUTESCLOUTKERNELDEGREE Compute scaled kernel degree for the OS data 
% of an nlsaModel_scl object 
% 
% Modified 2014/07/28

sDistance = getSclOutSymmetricDistance( obj );
diffOp    = getSclOutDiffusionOperator( obj );

if nargin > 1 && ~isempty( iProc )
    nB         = getNTotalEmbBatch( obj );
    pPartition = nlsaPartition( 'nSample', nB, ...
                                'nBatch',  nProc );
    iBLim      = getBatchLimit( pPartition, iProc );
    varargin = [ varargin { 'batch' iBLim( 1 ) : iBLim( 2 ) } ];
    logFile  = sprintf( 'dataD_%i-%i.log', iProc, nProc );
else
    logFile = 'dataD.log';
end

computeDegree( diffOp, sDistance, ...
               'logPath', getDegreePath( diffOp ), ...
               'logFile', logFile, ...
               varargin{ : } );
