function computeOseStateError( obj, iProc, nProc, varargin )
% Compute state error of the OSE target data relative to the reference data
%
% Modified 2014/07/24

if nargin == 1
    iProc = 1;
    nProc = 1;
end

Opt.ifWriteX = true;
Opt = parseargs( Opt, varargin{ : } );

errComponent = getOseErrComponent( obj );
refComponent = getOseRefComponent( obj );
oseComponent = getOseEmbComponent( obj );

partition = getPartition( oseComponent( 1, : ) );
nB = getNTotalBatch( partition );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataEX_%i-%i.log', iProc, nProc );

computeData( errComponent, oseComponent, refComponent...
               'batch', iBLim( 1 ) : iBLim( 2 ), ...
               'logPath', getDataPath( oseErrComponent( 1 ) ), ...
               'logFile', logFile, ...
               'ifWriteX', Opt.ifWriteX );
