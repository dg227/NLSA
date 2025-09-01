function u = getSpatialPatterns( obj, varargin )
% GETSPATIALPATTERNS  Get spatial patterns (projected data) from a vector of 
% nlsaProjectedComponent objects
%
% Modified 2015/10/13

u = getProjectedData( obj, varargin{ : } );
