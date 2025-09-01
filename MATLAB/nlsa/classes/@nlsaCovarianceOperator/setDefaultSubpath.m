function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subdirectories of an nlsaCovarianceOperator 
% object 
%
% Modified 2014/07/16

obj = setOperatorSubpath( obj, getDefaultOperatorSubpath( obj ) );
obj = setSingularValueSubpath( obj, getDefaultSingularValueSubpath( obj ) );
obj = setLeftSingularVectorSubpath( obj, ...
        getDefaultLeftSingularVectorSubpath( obj ) );
obj = setRightSingularVectorSubpath( obj, ...
        getDefaultRightSingularVectorSubpath( obj ) );
