function computeOseDenKernelDoubleSum( obj, iProc, nProc, ...
                                             iDProc, nDProc, varargin )
% COMPUTEOSEDENKERNELDOUBLESUM Compute kernel double sum for the OSE density
% data of nlsaModel_den_ose objects  
% 
% Modified 2018/07/04

if nargin == 1
    iProc = 1;
    nProc = 1;
end

if nargin <= 3
    iDProc = 1;
    nDProc = 1;
end


pDistance = getOseDenPairwiseDistance( obj );
den       = getOseDensityKernel( obj );
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
    computeDoubleSum( den, pDistance, ...
                      'logPath', getDensityPath( den ), ...
                      'logFile', logFile, ...
                      varargin{ : } );
end


