function xiE2 = getRelativeVelocityNorm( obj, varargin )
% GETRELATIVEVELOCITYNORM Get squared relative space velocity 
% difference from an nlsaEmbeddedComponent_xi_d object
%
% Modified 2014/07/10

[ xiE2, xi2 ] = getVelocityNorm( obj, varargin{ : } );

xiE2 = xiE2 ./ xi2;

