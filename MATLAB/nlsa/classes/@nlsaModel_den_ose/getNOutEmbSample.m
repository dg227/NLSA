function nS = getNOutEmbSample( obj )
% GETNOUTEMBSAMPLE Get number of embedded OSE samples
%
% Modified 2018/07/01

nS = sum( getNSample( obj.outEmbComponent( 1, : ) ) );
