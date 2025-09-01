function [ zeta, mu, gamma ] = getEigenfunctions( obj, iB, iR, idxZeta, ...
                                                  varargin )
% GETEIGENFUNCTIONS  Read eigenfunction data of an nlsaKoopmanOperator 
% object
%
% Modified 2020/04/11

partition  = getPartition( obj );

if nargout >= 2 
    varNames = { 'zeta' 'mu'  };
else
    varNames = { 'zeta' };
end
file = fullfile( getEigenfunctionPath( obj ), getEigenfunctionFile( obj ) );
load( file, varNames{ : } )

if nargin < 4 || isempty( idxZeta )
    idxZeta = 1 : getNEigenfunction( obj );
end

if nargout == 3
    gamma = getEigenvalues( obj );
end


if nargin == 1 || isempty( iB )
    zeta = zeta( :, idxZeta );
    return
end

if nargin < 3 || isempty( iR ) 
    iBG = iB;
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

partitionG = mergePartitions( partition );
zeta = zeta( :, idxZeta );

if isscalar( iBG )
    iS = getBatchLimit( partitionG, iBG );
    zeta = zeta( iS( 1 ) : iS( 2 ), : );
    if nargout > 1
        mu = mu( iS( 1 ) : iS( 2 ) );
    end
else
    isContiguous = all( iBG( 2 : end ) - iBG( 1 : end - 1 ) == 1 );
    if isContiguous
        iSB1 = getBatchLimit( partitionG, iBG( 1 ) );
        iSB2 = getBatchLimit( partitionG, iBG( end ) );
        zeta = zeta( iSB1( 1 ) : iSB2( 2 ), : );
        if nargout > 1 
            mu = mu( iSB1( 1 ) : iSB2( 2 ) );
        end
    else
        nS = sum( getBatchSize( partitionG, iBG ) );
        zetaOut = zeros( nS, getNEigenfunction( obj ) );
        if nargout > 1
            muOut = zeros( nS, 1 );
        end
        iS1 = 1;
        for i = 1 : numel( iBG )
            iS2 = iS1 + getBatchSize( partitionG, iBG ) - 1;
            iSB = getBatchLimit( partitionG, iBG );
            zetaOut( iS1 : iS2, : ) = zeta( iSB( 1 ) : iSB( 2 ), : );
            if nargout > 1
                muOut( iS1 : iS2 ) = mu( iSB( 1 ) : iSB( 2 ) );
            end
            iS1 = iS2 + 1;
        end
        zeta = zetaOut;
        if nargout > 1 
            mu = muOut;
        end
    end
end

