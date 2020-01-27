function [ nDE, nSB ] = getBatchArraySize( obj, iB )
% GETBATCHARRAYSIZE  Get batch sizes of an nlsaComponent object 
%
% Modified 2020/01/25

if ~isscalar( obj )
    error( 'First argument must be a scalar nlsaEmbeddedComponent_e object.' )
end


if nargin == 1
    iB = 1 : getNBatch( obj );
end

nDE = getDataSpaceDimension( obj );
nSB = getBatchSize( obj, iB );

