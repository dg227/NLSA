function lim = getBatchLimit( obj, iB, iR );
% GETBATCHLIMIT  Get batch limits of nlsaPairwiseDistance object 
%
% Modified 2014/01/04

if nargin == 1
    iR = 1;
end

lim = getBatchLimit( obj.partition( iR ), iB );
