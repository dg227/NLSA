function nS = getNSrcSample( obj )
% GETNEMBSAMPLE Get number of embedded samples
%
% Modified 2014/07/22

cmp = getSrcComponent( obj );
nS = getNSample( cmp( 1, : ) );
