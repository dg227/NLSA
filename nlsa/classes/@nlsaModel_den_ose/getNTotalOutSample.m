function nS = getNTotalOutSample( obj )
% GETNTOTALOUTSAMPLE Get number of source samples
%
% Modified 2019/07/06

cmp = getOutComponent( obj );
nS = getNTotalSample( cmp( 1, : ) );
