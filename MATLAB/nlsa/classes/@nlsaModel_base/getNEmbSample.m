function nS = getNEmbSample( obj )
% GETNEMBSAMPLE Get number of embedded samples
%
% Modified 2014/07/22

cmp = getEmbComponent( obj );
nS = getNSample( cmp( 1, : ) );
