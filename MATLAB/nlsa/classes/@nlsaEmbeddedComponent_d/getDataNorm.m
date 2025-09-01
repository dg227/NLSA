function xENorm2 = getDataNorm( obj, iB, iR )
% GETDATANORM  Read differenced data norm from an nlsaEmbeddedComponent_d 
% object
%
% Modified 2014/07/10


if ~isrow( obj )
    error( 'First input argument must be a row vector' )
end

partition = getPartition( obj( 1, : ) );
nBTot     = getNTotalBatch( partition );

if nargin < 2
    iB = 1 : nBTot;
end

if nargin < 3
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
end

varNames = { 'xENorm2' };

if isscalar( iB )
    file = fullfile( getDataPath( obj( iR ) ), ...
                     getDataFile( obj( iR ), iB ) );
    load( file, varNames{ 1 : nargout } )
else
    partitionG = mergePartitions( partition );
    nS          = sum( getBatchSize( partitionG, iBG ) );
    xENorm2    = zeros( 1, nS );
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getDataPath( obj( iR( i ) ) ), ...
                         getDataFile( obj( iR( i ) ), iB( i ) ) );
        B = load( file, varNames{ : } );
        xENorm2( iS1 : iS2 ) = B.xENorm2;
        iS1 = iS2 + 1;
    end
end
xENorm2 = sqrt( xENorm2 );
