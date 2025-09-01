function makeIsrEmbedding( obj, iProc, nProc, varargin )
% MAKEOSEEMBEDDING Perform out-of-sample extension (OSE) of the lagged-embedded
% data in an nlsaModel_ose object
%
% Modified 2014/07/24

if nargin == 1
    iProc = 1;
    nProc = 1;
end


oseComponent = getIsrEmbComponent( obj );
switch class( oseComponent )
    case 'nlsaEmbeddedComponent_ose' 
        embComponent = getOutTrgEmbComponent( obj );
    case 'nlsaEmbeddedComponent_ose_n'
        embComponent = getOutPrjComponent( obj );
end
oseDiffOp    = getIsrDiffusionOperator( obj );

partition = getPartition( oseDiffOp );
nB = getNTotalBatch( partition );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );
iBLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataX_%i-%i.log', iProc, nProc );

makeEmbedding( oseComponent, embComponent, oseDiffOp, ...
               'batch', iBLim( 1 ) : iBLim( 2 ), ...
               'logPath', getDataPath( oseComponent( 1 ) ), ...
               'logFile', logFile );
