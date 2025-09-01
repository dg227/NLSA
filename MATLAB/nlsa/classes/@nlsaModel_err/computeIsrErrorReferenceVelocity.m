function computeIsrErrorReferenceVelocity( obj, iProc, nProc, varargin )
% COMPUTEISRERRORREFERENCEVELOCITY Compute reference error velocity for an 
% nlsaModel_err object
%
% Modified 2014/07/26

if nargin == 1
    iProc = 1;
    nProc = 1;
end


refComponent = getRefComponent( obj );
prjComponent = getPrjComponent( obj );
diffOp       = getDiffusionOperator( obj );


partition = getPartition( diffOp );
nB = getNTotalBatch( partition );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataX_%i-%i.log', iProc, nProc );

computeVelocity( refComponent, prjComponent, diffOp, ...
                 'batch', iBLim( 1 ) : iBLim( 2 ), ...
                 'logPath', getDataPath( refComponent( 1 ) ), ...
                 'logFile', logFile );
