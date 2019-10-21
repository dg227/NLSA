function [ v, mu, lambda ] = getEigenfunctions( obj, iB, iR, idxPhi, varargin )
% GETEIGENFUNCTIONS  Read eigenfunction data of an nlsaDiffusionOperator_batch 
% object
%
% Modified 2018/06/16

Opt.ifMu   = true; % return orthonormal basis wrt Riemannian measure
Opt = parseargs( Opt, varargin{ : } );

partition  = getPartition( obj );
nBTot      = getNTotalBatch( partition );

if nargout >= 2 || Opt.ifMu
    varNames = { 'v' 'mu' };
else
    varNames = { 'v' };
end

if nargin < 2 || isempty( iB )
    iB = 1 : nBTot;
end

if nargin < 3 || isempty( iR )
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

if nargin < 4 || isempty( idxPhi )
    idxPhi = 1 : getNEigenfunction( obj );
end

if isscalar( iB )
    file = fullfile( getEigenfunctionPath( obj ), ...
                     getEigenfunctionFile( obj, iB, iR ) );
    load( file, varNames{ : } )
    v = v( :, idxPhi );
else
    partitionG = mergePartitions( partition );
    nS = sum( getBatchSize( partitionG, iBG ) );
    v = zeros( nS, numel( idxPhi ) );
    if nargout > 1 || Opt.ifMu
        mu = zeros( nS, 1 );
    end
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getEigenfunctionPath( obj ), ...
                         getEigenfunctionFile( obj, iB( i ), iR( i ) ) );
        B = load( file, varNames{ : } );
        v( iS1 : iS2, : ) = B.v( :, idxPhi );
        if nargout > 1 || Opt.ifMu
            mu( iS1 : iS2 ) = B.mu;
        end
        iS1 = iS2 + 1;
    end
end

if Opt.ifMu 
    v = bsxfun( @rdivide, v, sqrt( mu ) );
end

if nargout == 3
    lambda = getEigenvalues( obj );
end
