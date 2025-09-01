function computeIsrStateError( obj, iProc, nProc, varargin )
% COMPUTEMODOSESTATEERROR Compute ISR state error of an nlsaModel_err object
%
% Modified 2014/07/28

if nargin == 1
    iProc = 1;
    nProc = 1;
end

Opt.ifWriteX = true;
Opt = parseargs( Opt, varargin{ : } );

errComponent = getModErrComponent( obj );
refComponent = getIsrRecComponent( obj );
isrComponent = getIsrEmbComponent( obj );

partition = getPartition( modOseDiffOp );
nB = getNTotalBatch( partition );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataEX_%i-%i.log', iProc, nProc );

computeData( errComponent, isrComponent, refComponent, ...
             'batch', iBLim( 1 ) : iBLim( 2 ), ...
             'logPath', getiDataPath( errComponent( 1 ) ), ...
             'logFile', logFile, ...
             'ifWriteX', Opt.ifWriteX );
