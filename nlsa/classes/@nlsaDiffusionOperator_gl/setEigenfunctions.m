function setEigenfunctions( obj, v, mu, varargin )
% SETEIGENFUNCTIONS  Set eigenfunction data of an nlsaDiffusionOperator_batch
% object
%
% Modified 2014/07/22


if size( v, 1 ) ~= getNTotalSample( obj  )
    error( 'Incompatible number of samples' )
end
if size( v, 2 ) ~= getNEigenfunction( obj )
    error( 'Incompatible number of eigenfunctions' )
end
if ~isvector( mu ) || numel( mu ) ~= size( v, 1 )
    error( 'Invalid Riemannian measure' )
end

file = fullfile( getEigenfunctionPath( obj ), ... 
                 getEigenfunctionFile( obj ) );
save( file, 'v', 'mu', varargin{ : } )

