function nE = getDenEmbeddingWindow( obj, idxC )
% GETDENEMBEDDINGWINDOW  Returns the  embedding window (number of lags)  of the 
% density data of an nlsaModel_den object

% Modified 2014/12/30

if nargin == 1
    idxC = 1 : getNDenComponent( obj ) ;
end

nE = getEmbeddingWindow( obj.denEmbComponent( idxC, 1 ) ); 
