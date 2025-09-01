function computeOseDiffusionEigenfunctions( obj, varargin )
% COMPUTEOSEDIFFUSIONEIGENFUNCTIONS Compute out-of-sample extension (OSE) 
% eigenfunctions of nlsaModel_den_ose 
% 
% Modified 2018/07/04


diffOp    = getDiffusionOperator( obj );
oseDiffOp = getOseDiffusionOperator( obj );

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

computeEigenfunctions( oseDiffOp, diffOp, ...
                       'logPath', getEigenfunctionPath( oseDiffOp ), ...
                       'logFile', logFile, ...
                       varargin{ : } );

