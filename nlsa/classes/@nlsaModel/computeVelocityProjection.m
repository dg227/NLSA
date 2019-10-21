function computeVelocityProjection( obj, iProc, nProc, varargin )
% COMPUTEVELOCITYPROJECTION Compute projection of the target phase space 
% velocity data onto the diffusion eigenfunctions of an nlsaModel object
%
% Modified 2014/06/25

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

computeVelocityProjection( prjComponent, trgEmbComponent, diffOp, ...
                           'component', iCLim( 1 ) : iCLim( 2 ), ...
                           'logPath', getVelocityProjectionPath( prjComponent( 1 ) ), ...
                           'logFile', logFile );
