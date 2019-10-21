function [ xiENorm2, xiE, xiNorm2 ] = getVelocity( obj, iB, iR )
% GETVELOCITYERROR  Read phase space velocity error from an 
% nlsaEmbeddedComponent_ose object
%
% Modified 2014/05/26

partition = getPartition( obj( 1, : ) );
nBTot     = getNTotalBatch( partition );

if nargin < 2
    iB = 1 : nBTot;
end

if nargin < 3
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
end


varNames = { 'xiENorm2' 'xiE' 'xiNorm2' };

if isscalar( iB )
    file = fullfile( getVelocityPath( obj( iR ) ), ...
                     getVelocityFile( obj( iR ), iB ) );
    load( file, varNames{ 1 : nargout } )
else
    partitionG = mergePartitions( partition );
    nS          = sum( getBatchSize( partitionG, iBG ) );
    xiENorm2    = zeros( 1, nS );
    if nargout > 1
        nD      = getEmbeddingSpaceDimension( obj );
        xiE     = zeros( nD, nS );
    end
    if nargout > 2
        xiNorm2 = zeros( 1, nS );
    end
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getVelocityPath( obj( iR( i ) ) ), ...
                         getVelocityFile( obj( iR( i ) ), iB( i ) ) );
        B = load( file, varNames{ : } );
        xiENorm2( iS1 : iS2 ) = B.xiENorm2;
        if nargout > 1
            xiE( :, iS1 : iS2 ) = B.xiE;
        end
        if nargout > 2
            xiNorm2( iS1 : iS2 ) = B.xiNorm2;
        end
        iS1 = iS2 + 1;
    end
end

