function computeIsrDiffusionEigenfunctions( obj, varargin )
% COMPUTEISRDIFFUSIONEIGENFUNCTIONS Compute ISR eigenfunctions of 
% nlsaModel_err objects
% 
% Modified 2014/07/28


diffOp    = getOutDiffusionOperator( obj );
isrDiffOp = getIsrDiffusionOperator( obj );

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
    logFile  = sprintf( 'dataPhi_%i-%i.log', iProc, nProc );
else
    logFile = 'dataPhi.log';
end

computeEigenfunctions( isrDiffOp, diffOp, ...
                       'logPath', getEigenfunctionPath( isrDiffOp ), ...
                       'logFile', logFile, ...
                       varargin{ : } );

