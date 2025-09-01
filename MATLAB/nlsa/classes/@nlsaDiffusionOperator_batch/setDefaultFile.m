function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaDiffusionOperator_batch object 
%
% Modified 2014/04/09

obj = setDefaultFile@nlsaDiffusionOperator( obj );
obj = setNormalizationFile( obj, getDefaultNormalizationFile( obj ) );
obj = setDegreeFile( obj, getDefaultDegreeFile( obj ) );
