function nS = getNTotalOutEmbSample( obj )
% GETNTOTALOUTEMBSAMPLE Get number of out-of-sample embedded samples 
%
% Modified 2020/08/01

cmp = getOutEmbComponent( obj );
nS = getNTotalSample( cmp( 1, : ) );
