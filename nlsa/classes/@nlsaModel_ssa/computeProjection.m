function computeProjection( obj, iProc, nProc, varargin )
% COMPUTEPROJECTION Compute projection of the target data onto the covariance
% eigenfunctions of an nlsaModel_ssa object
%
% Modified 2016/05/31

if nargin == 1
    iProc = 1;
    nProc = 1;
end

%Opt.ifWriteXi = true;
%Opt = parseargs( Opt, varargin{ : } );

trgEmbComponent = getTrgEmbComponent( obj );
prjComponent    = getPrjComponent( obj );
covOp           = getCovarianceOperator( obj);
nCT = size( trgEmbComponent, 1 );


pPartition = nlsaPartition( 'nSample', nCT, ...
                            'nBatch',  nProc );
iCLim   = getBatchLimit( pPartition, iProc );
logFile = sprintf( 'dataA_%i-%i.log', iProc, nProc );

computeProjection( prjComponent, trgEmbComponent, covOp, ...
                   'component', iCLim( 1 ) : iCLim( 2 ), ...
                   'logPath', getProjectionPath( prjComponent( 1 ) ), ...
                   'logFile', logFile );
