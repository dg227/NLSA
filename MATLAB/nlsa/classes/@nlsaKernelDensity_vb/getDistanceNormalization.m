function rho = getDistanceNormalization( obj, iB, iR )
% GETDISTANCENORMALIZATION Read distance normalization data of an nlsaKernelDensity_vb object
%
% Modified 2015/12/15

partition  = getPartition( obj );

varNames = { 'rho'  };
file = fullfile( getDensityPath( obj ), getDistanceNormalizationFile( obj ) );

load( file, varNames{ : } )

if nargin == 1 || isempty( iB )
    return
end

if nargin < 3 || isempty( iR ) 
    iBG = iB;
    [ iB, iR ] = loc2gl( partition, iB, iR ); 
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

if isscalar( iB )
    iS = getBatchLimit( partition( iR ), iB );
    rho = rho( iS( 1 ) : iS( 2 ) );
else
    partitionG = mergePartitions( partition );
    isContiguous = all( iBG( 2 : end ) - iBG( 1 : end - 1 ) == 1 );
    if isContiguous
        iSB1 = getBatchLimit( partitionG, iBG( 1 ) );
        iSB2 = getBatchLimit( partitionG, iBG( end ) );
        rho = rho( iSB1( 1 ) : iSB2( 2 ) );
    else
        nS = sum( getBatchSize( partitionG, iBG ) );
        rhoOut = zeros( nS, 1 );
        iS1 = 1;
        for i = 1 : numel( iBG )
            iS2 = iS1 + getBatchSize( partitionG, iBG ) - 1;
            iSB = getBatchLimit( partitionG, iBG );
            rhoOut( iS1 : iS2 ) = rho( iSB( 1 ) : iSB( 2 ) );
            iS1 = iS2 + 1;
        end
        rho = rhoOut;
    end
end

