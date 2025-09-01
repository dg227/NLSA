function path = getNormalizationPath( obj, path )
% GETNORMALIZATIONPATH  Get kernel normalization path of an 
% nlsaDiffusionOperator_batch object
%
% Modified 2014/02/09

path = fullfile( getPath( obj ), getNormalizationSubpath( obj ) );
