function nS = getNTotalSample( obj )
% GETNTOTALSAMPLE  Get the total number of samples in an array of 
% nlsaComponent objects
%
% Modified  2014/07/22

nS = getNSample( obj );
nS = sum( nS( : ) );
