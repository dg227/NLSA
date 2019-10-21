function path = getProjectionPath( obj )
% GETPROJECTIONPATH  Get projection path of an nlsaProjectedComponent object
%
% Modified 2014/06/20

path = fullfile( getPath( obj ), getProjectionSubpath( obj ) );
