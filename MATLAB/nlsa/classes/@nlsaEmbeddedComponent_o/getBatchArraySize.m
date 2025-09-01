function [ nDE, nSB ] = getBatchArraySize( obj, iB )
% GETBATCHARRAYSIZE  Get batch sizes of an nlsaEmbeddedComponent_o object 
%
% Modified 2014/05/14

if nargin == 1
    iB = 1 : getNBatch( obj );
end
nDE = getDimension( obj );
nSB = getBatchSize( obj, iB );
nSB = nSB + getEmbeddingWindow( obj ) - 1;
