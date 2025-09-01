function nDE = getDenEmbeddingSpaceDimension( obj, idxC )
% GETDENEMBEDDINGSPACEDIMENSION  Get embedding space dimension of the 
% density data of an nlsaModel_den object

% Modified 2014/12/30

if nargin == 1
    idxC = 1 : getNDenComponent( obj ) ;
end

nDE = getEmbeddingSpaceDimension( obj.denEmbComponent( idxC, 1 ) ); 
