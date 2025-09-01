function nS = getNTotalSample( obj )
% GETNTOTALSAMPLE  Get total number of samples in an array of 
% nlsaPairwiseDistance objects
%
% Modified 2019/11/05

nS = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nS( iObj ) = getNTotalSample( obj( iObj ).partition );
end
