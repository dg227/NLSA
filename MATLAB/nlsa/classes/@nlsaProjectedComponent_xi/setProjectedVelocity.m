function setProjectedVelocity( obj, b, varargin )
% SETPROJECTEDVELOCITY  Set projected velocity data of an 
% nlsaProjectedComponent_xi object
%
% varargin is used to pass flags for Matlab's save function 
%
% Modified 2014/06/24

nDE = getEmbeddingSpaceDimension( obj );
nL  = getNBasisFunction( obj );

if any( size( b ) ~= [ nDE nL + 1 ] )
    error( 'Incompatible size of data array' )
end
    
file = fullfile( getVelocityProjectionPath( obj ), ... 
                 getVelocityProjectionFile( obj ) );

save( file, 'b', varargin{ : } )
