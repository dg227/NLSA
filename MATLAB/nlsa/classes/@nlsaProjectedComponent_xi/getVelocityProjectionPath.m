function path = getVelocityProjectionPath( obj )
% GETVELOCITYPROJECTIONPATH  Get velocity projection path of an 
% nlsaProjectedComponent_xi object
%
% Modified 2014/06/24

path = fullfile( getPath( obj ), getVelocityProjectionSubpath( obj ) );
