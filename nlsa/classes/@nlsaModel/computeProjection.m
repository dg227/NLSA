function computeProjection( obj, iProc, nProc, varargin )
% COMPUTEPROJECTION Compute projection of the target data onto the diffusion
% eigenfunctions of an nlsaModel object
%
% Modified 2014/06/24

if nargin == 1
    iProc = 1;
    nProc = 1;
end

%Opt.ifWriteXi = true;
%Opt = parseargs( Opt, varargin{ : } );

trgEmbComponent = getTrgEmbComponent( obj );
prjComponent    = getPrjComponent( obj );
diffOp          = getDiffusionOperator( obj);
nCT = size( trgEmbComponent, 1 );


pPartition = nlsaPartition( 'nSample', nCT, ...
                            'nBatch',  nProc );
iCLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataA_%i-%i.log', iProc, nProc );

computeProjection( prjComponent, trgEmbComponent, diffOp, ...
                   'component', iCLim( 1 ) : iCLim( 2 ), ...
                   'logPath', getProjectionPath( prjComponent( 1 ) ), ...
                   'logFile', logFile );
