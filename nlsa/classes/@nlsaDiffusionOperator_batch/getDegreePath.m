function path = getDegreePath( obj, path )
% GETDEGREEPATH  Get kernel degree path of an 
% nlsaDiffusionOperator_batch object
%
% Modified 2014/02/09

path = fullfile( getPath( obj ), getDegreeSubpath( obj ) );
