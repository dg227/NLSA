function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaProjectedData_xi object
%
% Modified 2014/06/24

obj = setDefaultFile@nlsaProjectedComponent( obj );
file = getDefaultVelocityProjectionFile( obj );
obj  = setVelocityProjectionFile( obj, file );
