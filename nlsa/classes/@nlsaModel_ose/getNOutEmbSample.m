function nS = getNOutEmbSample( obj )
% GETNOUTEMBSAMPLE Get number of embedded OSE samples
%
% Modified 2014/02/06

nS = sum( getNSample( obj.outEmbComponent( 1, : ) ) );
