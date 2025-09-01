function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaEmbeddedComponent_xi object 
%
% Modified 2014/04/04

obj = setDefaultFile@nlsaEmbeddedComponent( obj );
obj = setVelocityFile( obj, getDefaultVelocityFile( obj ) );

