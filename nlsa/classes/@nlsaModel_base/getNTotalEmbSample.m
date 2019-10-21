function nS = getNTotalEmbSample( obj )
% GETNEMBSAMPLE Get total number of embedded samples
%
% Modified 2014/07/22

cmp = getEmbComponent( obj );
nS = getNTotalSample( cmp( 1, : ) );
