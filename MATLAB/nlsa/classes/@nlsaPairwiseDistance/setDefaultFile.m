function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaPairwiseDistance object 
%
% Modified 2014/04/02

obj  = setDistanceFile( obj,  getDefaultDistanceFile( obj ) );
