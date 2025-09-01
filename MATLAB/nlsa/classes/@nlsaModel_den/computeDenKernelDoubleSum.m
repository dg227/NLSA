function computeDenKernelDoubleSum( obj, iProc, nProc, ...
                                             iDProc, nDProc, varargin )
% COMPUTEDENKERNELDOUBLESUM Compute kernel double sum for the density
% data of nlsaModel_den objects  
% 
% Modified 2019/11/05

if nargin == 1
    iProc = 1;
    nProc = 1;
end

if nargin <= 3
    iDProc = 1;
    nDProc = 1;
end


pDistance = getDenPairwiseDistance( obj );
den       = getDensityKernel( obj );
nB        = getNTotalBatch( getPartition( den( 1 ) ) );
nD        = numel( pDistance );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );

dPartition = nlsaPartition( 'nSample', nD, ...
                            'nBatch', nDProc );

iBLim = getBatchLimit( pPartition, iProc );
iDLim = getBatchLimit( dPartition, iDProc );


logFile = sprintf( 'dataRho_%i-%i.log', iBLim( 1 ), iBLim( 2 ) );

for iD = iDLim( 1 ) : iDLim( 2 )
    computeDoubleSum( den( iD ), pDistance( iD ), ...
                      'logPath', getDensityPath( den( iD ) ), ...
                      'logFile', logFile, ...
                      varargin{ : } );
end


