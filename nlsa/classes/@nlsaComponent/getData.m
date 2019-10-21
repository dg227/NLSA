function x = getData( obj, iB, iR, iC, iA )
% GETDATA  Read data from an nlsaComponent object
%
% Modified 2019/07/20

partition  = getPartition( obj );
nBTot      = getNTotalBatch( partition( 1, : ) );

if nargin < 2 || isempty( iB )
    iB = 1 : nBTot;
end

if nargin < 3 || isempty( iR )
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

if nargin < 4 || isempty( iC )
    iC = 1 : size( obj, 1 );
end

if nargin < 5 || isempty( iA )
    iA = 1;
end

if numel( size( obj ) ) == 3
     obj = squeeze( obj( :, :, iA ) );
elseif iA > 1 && numel( size( obj ) ) == 2
     error( 'Invalid argument iA' )
end


varNames = { 'x' };
if isscalar( iB ) && isscalar( iC )
    file = fullfile( getDataPath( obj( iC, iR ) ), ...
                     getDataFile( obj( iC, iR ), iB ) ) ;
    load( file, varNames{ : } )
else
    partitionG = mergePartitions( partition );
    nS = sum( getBatchSize( partitionG, iBG ) );
    nD = getDimension( obj( iC, 1 ) );
    nDTot = sum( nD );
    x = zeros( nDTot, nS );
    iS1 = 1;
    for j = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( j ) ), iB( j ) ) - 1;
        iD1 = 1;
        for i = 1 : numel( iC )
            iD2 = iD1 + nD( iC ) - 1;
            file = fullfile( getDataPath( obj( iC( i ), iR( j ) ) ), ...
                             getDataFile( obj( iC( i ), iR( j ) ), iB( j ) ) );
            B = load( file, varNames{ : } );
            x( iD1 : iD2, iS1 : iS2 ) = B.x;
            iD1 = iD2 + 1;
        end
        iS1 = iS2 + 1;
    end
end

