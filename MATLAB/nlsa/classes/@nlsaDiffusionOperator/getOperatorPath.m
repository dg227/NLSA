function path = getOperatorPath( obj )
% GETOPERATORPATH  Get diffusion operator path of an nlsaDiffusionOperator 
% object
%
% Modified 2014/07/16

path = fullfile( getPath( obj ), getOperatorSubpath( obj ) );
