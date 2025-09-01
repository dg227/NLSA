function nE = getEmbeddingWindow( obj )
% GETEMBEDDINGINDICES  Get width of embedding window of an array of nlsaEmbeddedComponent objects
%
% Modified 2014/02/03

nE = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nE( iObj ) = obj( iObj ).idxE( end );
end
