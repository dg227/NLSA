function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subpath of an nlsaProjectedComponent object
%
% Modified 2014/06/23

obj = setProjectionSubpath( obj, getDefaultProjectionSubpath( obj ) );

