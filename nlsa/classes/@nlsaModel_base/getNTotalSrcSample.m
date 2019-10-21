function nS = getNTotalSrcSample( obj )
% GETNEMBSAMPLE Get number of source samples
%
% Modified 2014/07/22

cmp = getSrcComponent( obj );
nS = getNTotalSample( cmp( 1, : ) );
