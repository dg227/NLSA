function [ nDE, nSB ] = getBatchArraySize( obj, iB )
% GETBATCHARRAYSIZE  Get batch sizes of an nlsaEmbeddedComponent_e object 
%
% Modified 2014/05/02

if ~isscalar( obj )
    error( 'First argument must be a scalar nlsaEmbeddedComponent_e object.' )
end


if nargin == 1
    iB = 1 : getNBatch( obj );
end

nDE = getEmbeddingSpaceDimension( obj );
nSB = getBatchSize( obj, iB );

