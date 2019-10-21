function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subdirectories of an 
% nlsaDiffusionOperator_batch object 
%
% Modified 2014/04/09

obj = setDefaultSubpath@nlsaDiffusionOperator( obj );
obj = setNormalizationSubpath( obj, getDefaultNormalizationSubpath( obj ) );
obj = setDegreeSubpath( obj, getDefaultDegreeSubpath( obj ) );
