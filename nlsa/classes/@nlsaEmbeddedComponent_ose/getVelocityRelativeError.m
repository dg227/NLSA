function xiE2 = getVelocityRelativeError( obj, varargin )
% GETVELOCITYRELATIVEERROR Get squared relative space velocity error from an 
% nlsaEmbeddedComponent_ose object
%
% Modified 2014/04/24

[ xiE2, xi2 ] = getVelocityErrorNorm( obj, varargin{ : } );

xiE2 = xiE2 ./ xi2;

