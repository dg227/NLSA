function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaSymmetricDistance_gl object 
%
% Modified 2013/12/12

file = getDefaultDistanceFile( obj );
obj  = setDistanceFile( obj, file );
