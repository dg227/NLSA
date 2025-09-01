function nDE = getEmbeddingSpaceDimension( obj )
% GETEMBEDDINGSPACEDIMENSION  Get delay embedding space dimension of 
% nlsaLocalDistanceData object
%
% Modified  2015/10/26

comp = getComponent( obj );
nDE = getEmbeddingSpaceDimension( comp( :, 1 ) ); 
