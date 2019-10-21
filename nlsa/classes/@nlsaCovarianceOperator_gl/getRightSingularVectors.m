function [ v, s ] = getRightSingularVectors( obj, iB, iR, idxPhi )
% GETRIGHTSINGULARVECTORS  Read right singular vectors of an 
% nlsaCovarianceOperator_gl object
%
% Modified 2017/02/11

partition  = getPartition( obj );


varNames = { 'v' };
file = fullfile( getRightSingularVectorPath( obj ), ...
                 getRightSingularVectorFile( obj ) );

load( file, varNames{ : } )

if nargin < 4 || isempty( idxPhi )
    idxPhi = 1 : getNEigenfunction( obj );
end

if nargin == 1 || isempty( iB )
    v = v( :, idxPhi );
    return
end

if nargin < 3 || isempty( iR ) 
    iBG = iB;
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

partitionG = mergePartitions( partition );
v = v( :, idxPhi );

if isscalar( iBG )
    iS = getBatchLimit( partitionG, iBG );
    v = v( iS( 1 ) : iS( 2 ), : );
else
    isContiguous = all( iBG( 2 : end ) - iBG( 1 : end - 1 ) == 1 );
    if isContiguous
        iSB1 = getBatchLimit( partitionG, iBG( 1 ) );
        iSB2 = getBatchLimit( partitionG, iBG( end ) );
        v = v( iSB1( 1 ) : iSB2( 2 ), : );
    else
        nS = sum( getBatchSize( partitionG, iBG ) );
        vOut = zeros( nS, getNEigenfunction( obj ) );
        iS1 = 1;
        for i = 1 : numel( iBG )
            iS2 = iS1 + getBatchSize( partitionG, iBG ) - 1;
            iSB = getBatchLimit( partitionG, iBG );
            vOut( iS1 : iS2, : ) = v( iSB( 1 ) : iSB( 2 ), : );
            if nargout > 1
                muOut( iS1 : iS2 ) = mu( iSB( 1 ) : iSB( 2 ) );
            end
            iS1 = iS2 + 1;
        end
        v = vOut;
    end
end

if nargout == 2
    s = getSingularValues( obj );
end
