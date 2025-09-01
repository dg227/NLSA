function a = getLinearMap( obj )
% GETLINEARMAP  Read linear map of an nlsaCovarianceOperator_gl object
%
% Modified 2014/07/17


varNames = { 'a'  };
file = fullfile( getOperatorPath( obj ), getLinearMap( obj ) );
load( file, varNames{ : } )
