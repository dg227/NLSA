function minIdxO = getMinOrigin( obj )
% GETMINORIGIN  Get minimum time origin in an array of nlsaEmbeddedComponent objects
%
% Modified 2019/06/21

nE      = getEmbeddingWindow( obj ) + getNXB( obj );
minIdxO = min( nE( : ) );
