function computeOseVelocityError( obj, iProc, nProc, varargin )
% Compute phase space velocity error of OSE data relative to the source data
%
% Modified 2014/07/24

if nargin == 1
    iProc = 1;
    nProc = 1;
end

Opt.ifWriteXi = true;
Opt = parseargs( Opt, varargin{ : } );

errComponent = getOseErrComponent( obj );
refComponent = getOutRefComponent( obj );
oseComponent = getOseEmbComponent( obj );

partition = getPartition( oseComponent( 1, : ) );
nB = getNTotalBatch( partition );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataEXi_%i-%i.log', iProc, nProc );

computeVelocity( errComponent, oseComponent, refComponent...
                'batch', iBLim( 1 ) : iBLim( 2 ), ...
                'logPath', getDataPath( oseErrComponent( 1 ) ), ...
                'logFile', logFile, ...
                'ifWriteX', Opt.ifWriteXi );


