function computeOseDensity( obj, iProc, nProc, ...
                              iDProc, nDProc, varargin )
% COMPUTEOSEDENSITY Compute OSE kernel density estimate for nlsaModel_den_ose objects  
% 
% Modified 2018/07/07

if nargin == 1
    iProc = 1;
    nProc = 1;
end

if nargin <= 3
    iDProc = 1;
    nDProc = 1;
end

pDistance = getOseDenPairwiseDistance( obj );
oseDen    = getOseDensityKernel( obj );
den       = getDensityKernel( obj );
nB        = getNTotalBatch( getPartition( oseDen ) );
nD        = numel( oseDen );

pPartition = nlsaPartition( 'nSample', nB, ...
                            'nBatch',  nProc );

dPartition = nlsaPartition( 'nSample', nD, ...
                            'nBatch', nDProc );

iBLim = getBatchLimit( pPartition, iProc );
iDLim = getBatchLimit( dPartition, iDProc );

logFile = sprintf( 'dataRho_%i-%i.log', iProc, nProc );

for iD = iDLim( 1 ) : iDLim( 2 )
    computeDensity( oseDen( iD ), pDistance( iD ), den( iD ),  ...
                        'logPath', getDensityPath( oseDen( iD ) ), ...
                        'logFile', logFile, ...
                        varargin{ : } );
end
