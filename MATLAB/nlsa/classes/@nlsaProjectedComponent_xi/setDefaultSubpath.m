function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subpath of an nlsaProjectedComponent_xi object
%
% Modified 2014/06/24

obj = setDefaultSubpath@nlsaProjectedComponent( obj );
obj = setVelocityProjectionSubpath( obj, ...
    getDefaultVelocityProjectionSubpath( obj ) );

