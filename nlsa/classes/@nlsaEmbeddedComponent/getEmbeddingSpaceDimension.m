function nDE = getEmbeddingSpaceDimension( obj )
% GETEMBEDDINGSPACEDIMENSION  Get embedding space dimension of an
% array of nlsaEmbeddedComponent objects
%
% Modified 2013/04/15

nDE = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nDE( iObj ) = obj( iObj ).nD * numel( obj( iObj ).idxE );
end
