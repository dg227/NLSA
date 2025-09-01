function computeOutKernelDegree( obj, varargin )
% COMPUTEOUTKERNELDEGREE Compute kernel degree for OS data of nlsaModel_err 
% objects  
% 
% Modified 2014/05/25

sDistance = getOutSymmetricDistance( obj );
diffOp    = getOutDiffusionOperator( obj );

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
