function nE = getEmbeddingWindow( obj )
% GETEMBEDDINGWINDOW  Get embedding window of nlsaLocalDistanceData object
%
% Modified  2015/10/23

comp = getComponent( obj );
nE = getEmbeddingWindow( comp( 1, : ) ); 
