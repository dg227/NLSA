function setTemporalPatterns( obj, vT, mu, varargin )
% SETTEMPORALPATTERNS  Set temporal patterns of an nlsaLinearMap_gl object
%
% Modified 2015/10/20


if size( vT, 1 ) ~= getNTotalSample( obj  )
    error( 'Incompatible number of samples' )
end
if size( vT, 2 ) ~= getNEigenfunction( obj ) 
    error( 'Incompatible number of eigenfunctions' )
end
if ~isvector( mu ) || numel( mu ) ~= size( vT, 1 )
    error( 'Invalid Riemannian measure' )
end

file = fullfile( getTemporalPatternPath( obj ), ... 
                 getTemporalPatternFile( obj ) );
save( file, 'vT', 'mu', varargin{ : } )

