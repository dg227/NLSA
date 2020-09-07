function computeKoopmanProjection( obj, iC )
% COMPUTEKOOPMANPROJECTION Compute projection of the target data onto the 
% Koopman eigenfunctions of an nlsaModel object
%
% Modified 2020/08/31

trgEmbComponent = getTrgEmbComponent( obj );
prjComponent    = getKoopmanPrjComponent( obj );
kOp             = getKoopmanOperator( obj);

nCT = size( trgEmbComponent, 1 );

if nargin == 1
    iC = 1 : nCT;
end

logFile = sprintf( 'dataA_%s.log', idx2str( iC ) );

computeProjection( prjComponent, trgEmbComponent, kOp, ...
                   'component', iC, ...
                   'logPath', getProjectionPath( prjComponent( iC( 1 ) ) ), ...
                   'logFile', logFile );
