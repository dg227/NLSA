function xiNorm2 = getReferenceVelocityNorm( obj, iB, iR )
% GETREFERENCEVELOCITYNORM  Read phase space velocity norm of reference data 
% of an nlsaEmbeddedComponent_ose object
%
% Modified 2014/05/25


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

varNames = { 'xiNorm2' };

if isscalar( iB )
    file = fullfile( getVelocityErrorPath( obj( iR ) ), ...
                     getVelocityErrorFile( obj( iR ), iB ) );
    load( file, varNames{ 1 : nargout } )
else
    partitionG = mergePartitions( partition );
    nS          = sum( getBatchSize( partitionG, iBG ) );
    xiNorm2 = zeros( 1, nS );
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getVelocityErrorPath( obj( iR( i ) ) ), ...
                         getVelocityErrorFile( obj( iR( i ) ), iB( i ) ) );
        B = load( file, varNames{ : } );
        xiNorm2( iS1 : iS2 ) = B.xiNorm2;
        iS1 = iS2 + 1;
    end
end
xiNorm2 = sqrt( xiNorm2 );
