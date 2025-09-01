function nS = getNTotalOutEmbSample( obj )
% GETNTOTALOUTEMBSAMPLE Get number of out-of-sample embedded samples 
%
% Modified 2021/07/05

cmp = getOutEmbComponent( obj );
nS = getNTotalSample( cmp( 1, : ) );
