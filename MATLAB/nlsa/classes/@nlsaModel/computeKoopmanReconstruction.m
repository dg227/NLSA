function computeKoopmanReconstruction( obj, iB, iC, iR, iRec, nPar )
% COMPUTEKOOPMANRECONSTRUCTION Reconstruct the target data from the 
% projections onto the Koopman eigenfunctions of an nlsaModel object.
%
% Modified 2020/08/31

trgEmbComponent = getTrgEmbComponent( obj );
prjComponent    = getKoopmanPrjComponent( obj );
koopOp          = getKoopmanOperator( obj);
recComponent    = getKoopmanRecComponent( obj );

nCT  = size( recComponent, 1 );
nR   = size( recComponent, 2 );
nRec = size( recComponent, 3 );

if nargin < 6 
    nPar = 0;
end
if nargin < 5 || isempty( iRec )
    iRec = 1 : nRec;
end
if nargin < 4 || isempty( iR )
    iR = 1 : nR;
end
if nargin < 3 || isempty( iC )
    iC = 1 : nCT;
end

if nargin < 2 || isempty( iB )
    iB = 1 : getNBatch( getPartition( recComponent( iC, iR ) ) );
end

[ iC, iR, iRec ] = ndgrid( iC, iR, iRec );

logFile   = 'dataX_rec.log';

parfor( i = 1 : numel( iC ), nPar )
    pth = getDataPath( recComponent( iC( i ), iR( i ), iRec( i ) ) );
    computeData( recComponent( iC( i ), iR( i ), iRec( i ) ), ...
                 prjComponent, koopOp, ...
                 iC( i ), iR( i ), ...
                 'batch',   iB, ...
                 'logPath', pth, ... 
                 'logFile', logFile );
end
