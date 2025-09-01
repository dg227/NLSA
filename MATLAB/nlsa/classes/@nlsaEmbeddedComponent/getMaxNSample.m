function maxNSE = getMaxNSample( obj, nS )
% GETMAXNSAMPLE  Get maximum allowable number of samples in an array of nlsaEmbeddedComponent objects after emedding source data with nS samples
%
% Modified 2013/12/10

if any( size( obj ) ~= size( nS ) )
    error( 'Incompatible source sample array' )
end

maxNSE = min( nS - getOrigin( obj ) ) + 1;

