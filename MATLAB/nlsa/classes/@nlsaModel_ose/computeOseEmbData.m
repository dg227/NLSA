function computeOseEmbData( obj, iProc, nProc, varargin )
% COMPUTEOSEEMBDATA Perform out-of-sample extension (OSE) of the lagged-embedded
% data in an nlsaModel_ose object
%
% Modified 2015/12/02

if nargin == 1
    iProc = 1;
    nProc = 1;
end


oseComponent = getOseEmbComponent( obj );
switch class( oseComponent )
    case 'nlsaEmbeddedComponent_ose' 
        embComponent = getTrgEmbComponent( obj );
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

computeData( oseComponent, embComponent, oseDiffOp, ...
            'batch', iBLim( 1 ) : iBLim( 2 ), ...
            'logPath', getDataPath( oseComponent( 1 ) ), ...
            'logFile', logFile );
