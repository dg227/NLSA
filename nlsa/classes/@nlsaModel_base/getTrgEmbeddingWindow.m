function nE = geTrgtEmbeddingWindow( obj, idxC )
% GETTRGEMBEDDINGWINDOW  Returns the  embedding window (number of lags)  of the target data of an nlsaModel_base object

% Modified 2014/01/09

if nargin == 1
    idxC = 1 : getNTrgComponent( obj ) ;
end

nE = getEmbeddingWindow( obj.trgEmbComponent( idxC, 1 ) ); 
