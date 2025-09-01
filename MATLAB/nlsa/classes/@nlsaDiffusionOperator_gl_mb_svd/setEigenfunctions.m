function setEigenfunctions( obj, u, mu, v, varargin )
% SETEIGENFUNCTIONS  Set eigenfunction data of an 
% nlsaDiffusionOperator_gl_mb_svd object
%
% Modified 2018/06/14

nU = size( u );
nV = size( v );

if size( u, 1 ) ~= getNTotalSample( obj  )
    error( 'Incompatible number of samples' )
end
if size( u, 2 ) ~= getNEigenfunction( obj )
    error( 'Incompatible number of eigenfunctions' )
end
if ~isvector( mu ) || numel( mu ) ~= size( v, 1 )
    error( 'Invalid Riemannian measure' )
end
if any( nU ~= nV )
    error( 'Incompatible right singular vectors' )
end


file = fullfile( getEigenfunctionPath( obj ), ... 
                 getEigenfunctionFile( obj ) );
save( file, 'u', 'v', 'mu', varargin{ : } )

