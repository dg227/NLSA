function setProjectedData( obj, a, varargin )
% SETPROJECTEDDATA  Set projected data of an nlsaProjectedComponent object
%
% varargin is used to pass flags for Matlab's save function 
%
% Modified 2015/05/11

nDE = getEmbeddingSpaceDimension( obj );
nL  = getNBasisFunction( obj );

if any( size( a ) ~= [ nDE nL  ] )
    error( 'Incompatible size of data array' )
end
    
file = fullfile( getProjectionPath( obj ), ... 
                 getProjectionFile( obj ) );

save( file, 'a', varargin{ : } )
