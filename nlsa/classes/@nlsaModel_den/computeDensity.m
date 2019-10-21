function computeDensity( obj, iProc, nProc, ...
                              iDProc, nDProc, varargin )
% COMPUTEDENSITY Compute kernel density estimate for  nlsaModel_den objects  
% 
% Modified 2015/10/31

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
nB        = getNTotalBatch( getPartition( den ) );
nD        = numel( den );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );

dPartition = nlsaPartition( 'nSample', nD, ...
                            'nBatch', nDProc );

iBLim = getBatchLimit( pPartition, iProc );
iDLim = getBatchLimit( dPartition, iDProc );

logFile = sprintf( 'dataRho_%i-%i.log', iProc, nProc );

for iD = iDLim( 1 ) : iDLim( 2 )
    computeDensity( den( iD ), pDistance( iD ), ...
                        'logPath', getDensityPath( den( iD ) ), ...
                        'logFile', logFile, ...
                        varargin{ : } );
end
