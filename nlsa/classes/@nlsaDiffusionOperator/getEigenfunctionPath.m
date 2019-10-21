function path = getEigenfunctionPath( obj )
% GETOPERATORPATH  Get eigenfunction path of an nlsaDiffusionOperator 
% object
%
% Modified 2014/02/07

path = fullfile( getPath( obj ), getEigenfunctionSubpath( obj ) );
