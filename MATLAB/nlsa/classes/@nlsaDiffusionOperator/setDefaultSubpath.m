function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subdirectories of an nlsaDiffusionOperator 
% object 
%
% Modified 2014/04/03

obj = setOperatorSubpath( obj, getDefaultOperatorSubpath( obj ) );
obj = setEigenfunctionSubpath( obj, getDefaultEigenfunctionSubpath( obj ) );
