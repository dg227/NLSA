function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaEmbeddedComponent_ose object 
%
% Modified 2014/05/20

obj = setDefaultFile@nlsaEmbeddedComponent_xi_e( obj );
obj = setStateErrorFile( obj, getDefaultStateErrorFile( obj ) );
obj = setVelocityErrorFile( obj, getDefaultVelocityErrorFile( obj ) );

