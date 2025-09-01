function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subpath of nlsaEmbeddedComponent_ose object
%
% Modified 2014/05/25

obj = setDefaultSubpath@nlsaEmbeddedComponent_xi( obj );
obj = setStateErrorSubpath( obj, getDefaultStateErrorSubpath( obj ) );
obj = setVelocityErrorSubpath( obj, getDefaultVelocityErrorSubpath( obj ) );

