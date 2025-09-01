function q = getNormalization( obj, iB, iR )
% GETNORMALIZATION  Read normalization data of an nlsaDiffusionOperator_gl 
% object
%
% Modified 2015/01/04

file = fullfile( getOperatorPath( obj ), getOperatorFile( obj ) );
load( file, 'q' );

if nargin == 1 || isempty( iB )
    return
end

partition  = getPartition( obj );

if nargin < 3 || isempty( iR ) 
    iBG = iB;
    [ iB, iR ] = loc2gl( partition, iB, iR ); 
else
    iBG = loc2gl( partition, iB, iR );
end


if isscalar( iB )
    iS = getBatchLimit( partition( iR ), iB );
    q = q( iS( 1 ) : iS( 2 ) );
else
    partitionG = mergePartitions( partition );
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
            qOut( iS1 : iS2, : ) = q( iSB( 1 ) : iSB( 2 ) );
            iS1 = iS2 + 1;
        end
        q = qOut;
    end
end

