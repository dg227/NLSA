function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subdirectories of an nlsaLinearMap object
%
% Modified 2015/10/19

obj = setDefaultSubpath@nlsaCovarianceOperator_gl( obj );

obj = setTemporalPatternSubpath( obj, getDefaultTemporalPatternSubpath( obj ) );
