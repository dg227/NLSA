function nE = getEmbeddingWindow( obj, idxC )
% GETEMBEDDINGWINDOW  Returns the  embedding window (number of lags)  of the 
% source data of an nlsaModel_base object

% Modified 2014/01/09

if nargin == 1
    idxC = 1 : getNSrcComponent( obj ) ;
end

nE = getEmbeddingWindow( obj.embComponent( idxC, 1 ) ); 
