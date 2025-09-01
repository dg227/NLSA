function nR = getNRealization( obj )
% GETNREALIZATIONS  Get number of realizations of an nlsaPairwiseDistance object
%
% Modified  2014/01/06

nR = numel( obj.partition );
