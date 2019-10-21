function computeKernelDoubleSum( obj, iProc, nProc, varargin )
% COMPUTEKERNELDOUBLESUM Compute kernel double sum 
% data of nlsaModel objects  
% 
% Modified 2015/05/08

if nargin == 1
    iProc = 1;
    nProc = 1;
end

sDistance = getSymmetricDistance( obj );
diffOp    = getDiffusionOperator( obj );
nB        = getNTotalBatch( getPartition( diffOp ) );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim      = getBatchLimit( pPartition, iProc );

logFile = sprintf( 'dataRho_%i-%i.log', iProc, nProc );


%computeKernelNormalization( den, pDistance, ...
%                           'batch',   iBLim( 1 ) : iBLim( 2 ), ...
%                      'logPath', getNormalizationPath( diffOp ), ...
%                      'logFile', logFile, ...
%                      varargin{ : } );


computeDoubleSum( diffOp, sDistance, ...
                  'logPath', getOperatorPath( diffOp ), ...
                  'logFile', logFile, ...
                  varargin{ : } );
