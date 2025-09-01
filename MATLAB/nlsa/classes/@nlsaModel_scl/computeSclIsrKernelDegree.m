function computeSclIsrKernelDegree( obj, varargin )
% COMPUTESCLISRKERNELDEGREE Compute kernel degree of the scaled ISR 
% operator of an nlsaModel_scl object  
% 
% Modified 2014/07/28


pDistance = getSclOutPairwiseDistance( obj );
diffOp    = getSclIsrDiffusionOperator( obj );

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
    logFile  = sprintf( 'dataD_%i-%i.log', iProc, nProc );
else
    logFile = 'dataD_1_1.log';
end

computeDegree( diffOp, pDistance, ...
                      'logPath', getDegreePath( diffOp ), ...
                      'logFile', logFile, ...
                      varargin{ : } );
