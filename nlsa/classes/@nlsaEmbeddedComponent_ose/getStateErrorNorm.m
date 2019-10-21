function xENorm2 = getStateErrorNorm( obj, iB, iR )
% GETSTATEERRORNORM  Read state error norm from an nlsaEmbeddedComponent_ose 
% object
%
% Modified 2014/05/26


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
    file = fullfile( getStateErrorPath( obj( iR ) ), ...
                     getStateErrorFile( obj( iR ), iB ) );
    load( file, varNames{ 1 : nargout } )
else
    partitionG = mergePartitions( partition );
    nS          = sum( getBatchSize( partitionG, iBG ) );
    xENorm2    = zeros( 1, nS );
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getStateErrorPath( obj( iR( i ) ) ), ...
                         getStateErrorFile( obj( iR( i ) ), iB( i ) ) );
        B = load( file, varNames{ : } );
        xENorm2( iS1 : iS2 ) = B.xENorm2;
        iS1 = iS2 + 1;
    end
end
xENorm2 = sqrt( xENorm2 );
