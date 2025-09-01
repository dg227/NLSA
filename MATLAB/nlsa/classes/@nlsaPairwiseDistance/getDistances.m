function [ yVal, yInd ] = getDistances( obj, iB, iR )
% GETDISTANCES  Read distances and nearest-neighbor indices from an 
% nlsaPairwiseDistance object
%
% Modified 2017/07/21

partition  = getPartition( obj );
nBTot      = getNTotalBatch( partition );

if nargin < 2 || isempty( iB )
    iB = 1 : nBTot;
end

if nargin < 3 || isempty( iR )
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

pth = getDistancePath( obj );
switch nargout
    case 1
        varNames = { 'yVal' };
    case 2
        varNames = { 'yVal' 'yInd' };
end

if isscalar( iB ) 
    file = fullfile( pth, getDistanceFile( obj, iB, iR ) ) ;
    load( file, varNames{ : } )
else
    partitionG = mergePartitions( partition );
    nS = sum( getBatchSize( partitionG, iBG ) );
    nN = getNNeighbors( obj );
    yVal = zeros( nS, nN );
    if nargout == 2
        yInd = zeros( nS, nN, 'int32' );
    end
    iS1 = 1;
    for j = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( j ) ), iB( j ) ) - 1;
        file = fullfile( pth, getDistanceFile( obj, iB( j ), iR( j ) ) );
        B = load( file, varNames{ : } );
        yVal( iS1 : iS2, : ) = B.yVal;
        if nargout == 2
            yInd( iS1 : iS2, : ) = B.yInd;
        end
        iS1 = iS2 + 1;
    end
end
