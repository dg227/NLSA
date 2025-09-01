function computeOutProjection( obj, iProc, nProc, varargin )
% COMPUTEOUTPROJECTION Compute projection of the OS target data onto 
% the OS diffusion eigenfunctions of the imperfect model of an 
% nlsaModel_err object
%
% Modified 2014/06/25

if nargin == 1
    iProc = 1;
    nProc = 1;
end

%Opt.ifWriteXi = true;
%Opt = parseargs( Opt, varargin{ : } );

trgEmbComponent = getOutTrgEmbComponent( obj );
prjComponent    = getOutPrjComponent( obj );
diffOp          = getOutDiffusionOperator( obj);
nCT = size( trgEmbComponent, 1 );


pPartition = nlsaPartition( 'nSample', nCT, ...
                            'nBatch',  nProc );
iCLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataA_%i-%i.log', iProc, nProc );

computeProjection( prjComponent, trgEmbComponent, diffOp, ...
                   'component', iCLim( 1 ) : iCLim( 2 ), ...
                   'logPath', getProjectionPath( prjComponent( 1 ) ), ...
                   'logFile', logFile );
