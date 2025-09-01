function d = getDegree( obj, iB, iR )
% GETDEGREE Read degree data of an nlsaDiffusionOperator_batch object
%
% Modified 2014/04/22

partition = getPartition( obj );
nBTot     = getNTotalBatch( partition );

if nargin < 2
    iB = 1 : nBTot;
end

if nargin < 3
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
end

if isscalar( iB )
    file = fullfile( getDegreePath( obj ), ...
                     getDegreeFile( obj, iB, iR ) );
    load( file, 'd' )
else
    partitionG = mergePartitions( partition );
    nS = sum( getBatchSize( partitionG, iBG ) );
    d  = zeros( nS, 1 );
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getDegreePath( obj ), ...
                         getDegreeFile( obj, iB( i ), iR( i ) ) );
        B = load( file, 'd' );
        d( iS1 : iS2, : ) = B.d;
        iS1 = iS2 + 1;
    end
end
