function [ yVal, yInd ] = getDistances( obj, iB, iR )
% GETDISTANCES  Read distances and nearest-neighbor indices from an 
% nlsaSymmetricDistance_batch object
%
% Modified 2014/04/30

if nargin == 2
     iR = 1;
end

pth = getDistancePath( obj );
switch nargout
    case 1
        varList = { 'yVal' };
    case 2
        varList = { 'yVal' 'yInd' };
end

file = getDistanceFile( obj, iB, iR );
load( fullfile( pth, file ), varList{ : } )
