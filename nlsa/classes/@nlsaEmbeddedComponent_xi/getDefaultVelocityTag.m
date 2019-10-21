function tag = getDefaultVelocityTag( obj )
% GETDEFAULTVELOCITYTAG  Get default valocity tag of an 
% nlsaEmbeddedComponent_xi object
%
% Modified 2014/08/04

tag = sprintf( '%s_fdOrd%i', getFDType( obj ), getFDOrder( obj ) );
