function computeKoopmanProjection( obj, iProc, nProc, varargin )
% COMPUTEKOOPMANPROJECTION Compute projection of the target data onto the 
% Koopman eigenfunctions of an nlsaModel object
%
% Modified 2020/06/16

if nargin == 1
    iProc = 1;
    nProc = 1;
end

%Opt.ifWriteXi = true;
%Opt = parseargs( Opt, varargin{ : } );

trgEmbComponent = getTrgEmbComponent( obj );
prjComponent    = getKoopmanPrjComponent( obj );
kOp             = getKoopmanOperator( obj);
nCT = size( trgEmbComponent, 1 );


pPartition = nlsaPartition( 'nSample', nCT, ...
                            'nBatch',  nProc );
iCLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataA_%i-%i.log', iProc, nProc );

computeProjection( prjComponent, trgEmbComponent, kOp, ...
                   'component', iCLim( 1 ) : iCLim( 2 ), ...
                   'logPath', getProjectionPath( prjComponent( 1 ) ), ...
                   'logFile', logFile );
