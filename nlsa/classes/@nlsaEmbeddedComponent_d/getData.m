function [ xENorm2, xE ] = getData( obj, iB, iR )
% GETDATA  Read differenced data from an nlsaEmbeddedComponent_d object
%
% Modified 2014/07/10

partition = getPartition( obj( 1, : ) );
nBTot     = getNTotalBatch( partition );

if nargin < 2
    iB = 1 : nBTot;
end

if nargin < 3
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
end

varNames = { 'xENorm2' 'xE' };


if isscalar( iB )
    file = fullfile( getDataPath( obj( iR ) ), ...
                     getDataFile( obj( iR ), iB ) );
    load( file, varNames{ 1 : nargout } )
else
    partitionG = mergePartitions( partition );
    nS          = sum( getBatchSize( partitionG, iBG ) );
    xENorm2    = zeros( 1, nS );
    if nargout > 1
        nD      = getEmbeddingSpaceDimension( obj );
        xE     = zeros( nD, nS );
    end
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getDataPath( obj( iR( i ) ) ), ...
                         getDataFile( obj( iR( i ) ), iB( i ) ) );
        B = load( file, varNames{ : } );
        xENorm2( iS1 : iS2 ) = B.xENorm2;
        if nargout > 1
            xE( :, iS1 : iS2 ) = B.xE;
        end
        iS1 = iS2 + 1;
    end
end

