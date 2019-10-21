function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaCovarianceOperator_gl object
%
% Modified 2014/07/16

obj = setDefaultFile@nlsaCovarianceOperator( obj );

obj  = setRightSingularVectorFile( obj, ...
          getDefaultRightSingularVectorFile( obj ) );
obj = setLinearMapFile( obj, getDefaultLinearMapFile( obj ) );
obj = setRightCovarianceFile( obj, ...
          getDefaultRightCovarianceFile( obj ) );
obj = setLeftCovarianceFile( obj, ...
          getDefaultLeftCovarianceFile( obj ) );
