function computeOseVelocity( obj, iProc, nProc, varargin )
% Compute phase space velocity of OSE data
%
% Modified 2018/07/04

if nargin == 1
    iProc = 1;
    nProc = 1;
end


oseComponent = getOseEmbComponent( obj );
switch class( oseComponent )
    case 'nlsaEmbeddedComponent_ose' 
        embComponent = getEmbComponent( obj );
    case 'nlsaEmbeddedComponent_ose_n'
        embComponent = getPrjComponent( obj );
end
oseDiffOp    = getOseDiffusionOperator( obj );

partition = getPartition( oseDiffOp );
nB = getNTotalBatch( partition );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataX_%i-%i.log', iProc, nProc );

computeVelocity( oseComponent, embComponent, oseDiffOp, ...
                 'batch', iBLim( 1 ) : iBLim( 2 ), ...
                 'logPath', getDataPath( oseComponent( 1 ) ), ...
                 'logFile', logFile );

