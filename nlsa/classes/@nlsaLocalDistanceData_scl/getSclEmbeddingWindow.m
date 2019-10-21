function nE = getSclEmbeddingWindow( obj )
% GETSCLEMBEDDINGWINDOW  Get embedding window for the scaling data 
% of an nlsaLocalDistanceData object
%
% Modified  2015/10/23

comp = getSclComponent( obj );
nE = getEmbeddingWindow( comp( 1, : ) ); 
