function [ yRow, yCol, yVal ] = getDistances( obj )
% GETDATA  Read distance data from an nlsaSymmetricDistance object
%
% Modified 2014/04/16

varName = { 'yRow' 'yCol' 'yVal' };
load( fullfile( getDistancePath( obj ), getDistanceFile( obj ) ), ...
                varName{ 1 : nargout } )
