function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaLinearMap object
%
% Modified 2015/10/19

obj = setDefaultFile@nlsaCovarianceOperator_gl( obj );

obj  = setTemporalPatternFile( obj, ...
          getDefaultTemporalPatternFile( obj ) );
