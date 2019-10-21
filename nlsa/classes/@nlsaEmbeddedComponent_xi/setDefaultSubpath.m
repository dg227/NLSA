function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subpath of nlsaEmbeddedComponent_xi object
%
% Modified 2014/03/31

obj = setDefaultSubpath@nlsaEmbeddedComponent( obj );
obj = setVelocitySubpath( obj, getDefaultVelocitySubpath( obj ) );

