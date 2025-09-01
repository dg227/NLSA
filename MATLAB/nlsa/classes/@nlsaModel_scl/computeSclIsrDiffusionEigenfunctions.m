function computeSclIsrDiffusionEigenfunctions( obj, varargin )
% COMPUTESCLISRDIFFUSIONEIGENFUNCTIONS Compute ISR eigenfunctions with scaled
% kernel for an  nlsaModel_scl object 
% 
% Modified 2014/07/28


diffOp    = getOutDiffusionOperator( obj );
prjDiffOp = getSclIsrDiffusionOperator( obj );

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

computeEigenfunctions( prjDiffOp, diffOp, ...
                       'logPath', getEigenfunctionPath( prjDiffOp ), ...
                       'logFile', logFile, ...
                       varargin{ : } );

