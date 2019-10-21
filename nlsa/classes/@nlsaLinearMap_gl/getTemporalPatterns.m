function [ vT, mu, s ] = getTemporalPatterns( obj, iB, iR, idxPhi )
% GETTEMPORALPATTERNS  Read temporal patterns of an nlsaLinearMap_gl 
% object
%
% Modified 2017/02/11

partition  = getPartition( obj );

varNames = { 'vT' 'mu'  };
file = fullfile( getTemporalPatternPath( obj ), getTemporalPatternFile( obj ) );

load( file, varNames{ 1 : min( nargout, 2 ) } )

if nargin < 4 || isempty( idxPhi )
    idxPhi = 1 : getNEigenfunction( obj );
end

if nargout == 3
    s = getSingularValues( obj );
    s = s( idxPhi( idxPhi <= numel( s ) ) );
end

if nargin == 1 || isempty( iB )
    vT = vT( :, idxPhi );
    return
end

if nargin < 3 || isempty( iR ) 
    iBG = iB;
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

partitionG = mergePartitions( partition );
vT = vT( :, idxPhi );

if isscalar( iB )
    iS = getBatchLimit( partitionG( iBG ) );
    vT = vT( iS( 1 ) : iS( 2 ), : );
    if nargout > 1
        mu = mu( iS( 1 ) : iS( 2 ) );
    end
else
    isContiguous = all( iBG( 2 : end ) - iBG( 1 : end - 1 ) == 1 );
    if isContiguous
        iSB1 = getBatchLimit( partitionG, iBG( 1 ) );
        iSB2 = getBatchLimit( partitionG, iBG( end ) );
        vT = vT( iSB1( 1 ) : iSB2( 2 ), : );
        if nargout > 1 
            mu = mu( iSB1( 1 ) : iSB2( 2 ) );
        end
    else
        nS = sum( getBatchSize( partitionG, iBG ) );
        vTOut = zeros( nS, getNEigenfunction( obj ) );
        if nargout > 1
            muOut = zeros( nS, 1 );
        end
        iS1 = 1;
        for i = 1 : numel( iBG )
            iS2 = iS1 + getBatchSize( partitionG, iBG ) - 1;
            iSB = getBatchLimit( partitionG, iBG );
            vTOut( iS1 : iS2, : ) = vT( iSB( 1 ) : iSB( 2 ), : );
            if nargout > 1
                muOut( iS1 : iS2 ) = mu( iSB( 1 ) : iSB( 2 ) );
            end
            iS1 = iS2 + 1;
        end
        vT = vTOut;
        if nargout > 1 
            mu = muOut;
        end
    end
end

