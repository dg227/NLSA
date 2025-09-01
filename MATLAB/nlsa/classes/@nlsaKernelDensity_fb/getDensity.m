function q = getDensity( obj, iB, iR )
% GETDENSITY Read density data of an nlsaKernelDensity_fb object
%
% Modified 2020/04/04

partition  = getPartition( obj );

varNames = { 'q'  };
file = fullfile( getDensityPath( obj ), getDensityFile( obj ) );

load( file, varNames{ : } )

if nargin == 1 || isempty( iB )
    return
end

if nargin < 3 || isempty( iR ) 
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG ); 
elseif nargin == 3
    iBG = loc2gl( partition, iB, iR );
end

partitionG = mergePartitions( partition );

if isscalar( iBG )
    iS = getBatchLimit( partitionG, iBG );
    q = q( iS( 1 ) : iS( 2 ) );
else
    isContiguous = all( iBG( 2 : end ) - iBG( 1 : end - 1 ) == 1 );
    if isContiguous
        iSB1 = getBatchLimit( partitionG, iBG( 1 ) );
        iSB2 = getBatchLimit( partitionG, iBG( end ) );
        q = q( iSB1( 1 ) : iSB2( 2 ) );
    else
        nS = sum( getBatchSize( partitionG, iBG ) );
        qOut = zeros( nS, 1 );
        iS1 = 1;
        for i = 1 : numel( iBG )
            iS2 = iS1 + getBatchSize( partitionG, iBG ) - 1;
            iSB = getBatchLimit( partitionG, iBG );
            qOut( iS1 : iS2 ) = q( iSB( 1 ) : iSB( 2 ) );
            iS1 = iS2 + 1;
        end
        q = qOut;
    end
end

