function computeProjection( obj, iC )
% COMPUTEPROJECTION Compute projection of the target data onto the diffusion
% eigenfunctions of an nlsaModel object
%
% Modified 2020/08/31

trgEmbComponent = getTrgEmbComponent( obj );
prjComponent    = getPrjComponent( obj );
diffOp          = getDiffusionOperator( obj);

nCT = size( trgEmbComponent, 1 );

if nargin == 1
    iC = 1 : nCT;
end

logFile = sprintf( 'dataA_%i-%i.log', iProc, nProc );

computeProjection( prjComponent, trgEmbComponent, diffOp, ...
                   'component', iC, ...
                   'logPath', getProjectionPath( prjComponent( 1 ) ), ...
                   'logFile', logFile );
