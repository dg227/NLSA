function computeIsrVelocityError( obj, iProc, nProc, varargin )
% COMPUTEISRVELOCITYERROR Compute ISR phase space velocity error 
%
% Modified 2014/07/28

if nargin == 1
    iProc = 1;
    nProc = 1;
end

Opt.ifWriteXi = true;
Opt = parseargs( Opt, varargin{ : } );

errComponent = getModErrComponent( obj );
refComponent = getIsrRecComponent( obj );
isrComponent = getIsrEmbComponent( obj );

partition = getPartition( modOseDiffOp );
nB = getNTotalBatch( partition );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataEXi_%i-%i.log', iProc, nProc );

computeVelocity( errComponent, isrComponent, refComponent, ...
                'batch', iBLim( 1 ) : iBLim( 2 ), ...
                'logPath', getiDataPath( errComponent( 1 ) ), ...
                'logFile', logFile, ...
                'ifWriteXi', Opt.ifWriteXi );
