function [ v, mu, lambda ] = getEigenfunctions( obj, iB, iR, idxPhi, varargin )
% GETEIGENFUNCTIONS  Read eigenfunction data of an nlsaDiffusionOperator_gl 
% object
%
% Modified 2018/06/16

Opt.ifMu   = true; % return orthonormal basis wrt Riemannian measure
Opt = parseargs( Opt, varargin{ : } );


partition  = getPartition( obj );

if nargout >= 2 || Opt.ifMu
    varNames = { 'v' 'mu'  };
else
    varNames = { 'v' };
end
file = fullfile( getEigenfunctionPath( obj ), getEigenfunctionFile( obj ) );
load( file, varNames{ : } )

if nargin < 4 || isempty( idxPhi )
    idxPhi = 1 : getNEigenfunction( obj );
end

if nargout == 3
    lambda = getEigenvalues( obj );
end

if Opt.ifMu 
    v = bsxfun( @rdivide, v, sqrt( mu ) );
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
    if nargout > 1
        mu = mu( iS( 1 ) : iS( 2 ) );
    end
else
    isContiguous = all( iBG( 2 : end ) - iBG( 1 : end - 1 ) == 1 );
    if isContiguous
        iSB1 = getBatchLimit( partitionG, iBG( 1 ) );
        iSB2 = getBatchLimit( partitionG, iBG( end ) );
        v = v( iSB1( 1 ) : iSB2( 2 ), : );
        if nargout > 1 
            mu = mu( iSB1( 1 ) : iSB2( 2 ) );
        end
    else
        nS = sum( getBatchSize( partitionG, iBG ) );
        vOut = zeros( nS, getNEigenfunction( obj ) );
        if nargout > 1
            muOut = zeros( nS, 1 );
        end
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
        if nargout > 1 
            mu = muOut;
        end
    end
end

