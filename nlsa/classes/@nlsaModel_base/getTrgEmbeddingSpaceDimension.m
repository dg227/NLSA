function nDE = getTrgEmbeddingSpaceDimension( obj, idxC )
% GETEMBEDDINGSPACEDIMENSION  Get embedding space dimension of the target data 
% of an nlsaModel_base object

% Modified 2013/10/15

if nargin == 1
    idxC = 1 : getNSrcComponent( obj ) ;
end

nDE = getEmbeddingSpaceDimension( obj.trgEmbComponent( idxC, 1 ) ); 
