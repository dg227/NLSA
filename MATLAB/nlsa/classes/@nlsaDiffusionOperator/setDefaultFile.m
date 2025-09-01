function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaDiffusionOperator object 
%
% Modified 2014/04/08

obj  = setOperatorFile( obj, getDefaultOperatorFile( obj ) );
obj  = setEigenvalueFile( obj, getDefaultEigenvalueFile( obj ) );
obj  = setEigenfunctionFile( obj, getDefaultEigenfunctionFile( obj ) );
