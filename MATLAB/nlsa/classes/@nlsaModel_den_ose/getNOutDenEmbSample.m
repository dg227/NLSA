function nS = getNOutDenEmbSample( obj )
% GETNOUTDENEMBSAMPLE Get number of embedded OSE samples for density estimation
%
% Modified 2020/01/25

nS = sum( getNSample( obj.outDenEmbComponent( 1, : ) ) );
