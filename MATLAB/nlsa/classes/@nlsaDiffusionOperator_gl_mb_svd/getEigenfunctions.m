function [ u, mu, lambda, v ] = getEigenfunctions( obj, iB, iR, idxPhi, varargin )
% GETEIGENFUNCTIONS  Read eigenfunction data of an nlsaDiffusionOperator_gl_svd 
% object
%
% Modified 2018/06/14

Opt.ifMu   = true; % return orthonormal basis wrt Riemannian measure
Opt = parseargs( Opt, varargin{ : } );

partition  = getPartition( obj );
nSTot = getNTotalSample( obj );

file = fullfile( getEigenfunctionPath( obj ), getEigenfunctionFile( obj ) );
load( file, 'u', 'mu' )
if nargout == 4
    load( file, 'v' )
end

if nargin < 4 || isempty( idxPhi )
    idxPhi = 1 : getNEigenfunction( obj );
end

if nargout >= 3
    lambda = getEigenvalues( obj );
end

if Opt.ifMu 
    u = bsxfun( @rdivide, u, sqrt( mu ) );
    if nargout == 4
        v = bsxfun( @rdivide, v, sqrt( mu ) );
    end
end

if nargin == 1 || isempty( iB )
    u = u( :, idxPhi );
    if nargout == 4
        v = v( :, idxPhi );
    end
    return
end

if nargin < 3 || isempty( iR ) 
    iBG = iB;
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

partitionG = mergePartitions( partition );
u = u( :, idxPhi );
if nargout == 4
    v = v( :, idxPhi );
end

if isscalar( iBG )
    iS = getBatchLimit( partitionG, iBG );
    u = u( iS( 1 ) : iS( 2 ), : );
    if nargout > 1
        mu = mu( iS( 1 ) : iS( 2 ) );
    end
    if nargout == 4
        v = v( iS( 1 ) : iS( 2 ), : );
    end
else
    isContiguous = all( iBG( 2 : end ) - iBG( 1 : end - 1 ) == 1 );
    if isContiguous
        iSB1 = getBatchLimit( partitionG, iBG( 1 ) );
        iSB2 = getBatchLimit( partitionG, iBG( end ) );
        u = u( iSB1( 1 ) : iSB2( 2 ), : );
        if nargout > 1 
            mu = mu( iSB1( 1 ) : iSB2( 2 ) );
        end
        if nargout == 4
            v = v( iSB( 1 ) : iSB( 2 ), : );
        end
    else
        nS = sum( getBatchSize( partitionG, iBG ) );
        uOut = zeros( nS, getNEigenfunction( obj ) );
        if nargout > 1
            muOut = zeros( nS, 1 );
        end
        if nargout == 4
            vOut = zeros( nS, 1 );
        end
        iS1 = 1;
        for i = 1 : numel( iBG )
            iS2 = iS1 + getBatchSize( partitionG, iBG ) - 1;
            iSB = getBatchLimit( partitionG, iBG );
            uOut( iS1 : iS2, : ) = u( iSB( 1 ) : iSB( 2 ), : );
            if nargout > 1
                muOut( iS1 : iS2 ) = mu( iSB( 1 ) : iSB( 2 ) );
            end
            if nargout == 4
                vOut( iS1 : iS2, : ) = v( iSB( 1 ) : iSB( 2 ), : );
            end
            iS1 = iS2 + 1;
        end
        u = uOut;
        if nargout > 1 
            mu = muOut;
        end
        if nargout == 4
            v = vOut;
        end
    end
end

