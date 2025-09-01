function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaCovarianceOperator object
%
% Modified 2014/08/08

obj  = setSingularValueFile( obj, getDefaultSingularValueFile( obj ) );
obj  = setLeftSingularVectorFile( obj, ...
          getDefaultLeftSingularVectorFile( obj ) );
obj = setEntropyFile( obj, getSpectralEntropyFile( obj ) );
