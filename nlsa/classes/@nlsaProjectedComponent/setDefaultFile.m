function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaProjectedData object
%
% Modified 2014/06/20

file = getDefaultProjectionFile( obj );
obj  = setProjectionFile( obj, file );
