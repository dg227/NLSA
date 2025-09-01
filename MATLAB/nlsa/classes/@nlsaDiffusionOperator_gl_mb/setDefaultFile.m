function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaDiffusionOperator_gl_mb 
% object 
%
% Modified 2015/05/08

obj = setDefaultFile@nlsaDiffusionOperator_gl( obj );
obj  = setDoubleSumFile( obj, getDefaultDoubleSumFile( obj ) );
