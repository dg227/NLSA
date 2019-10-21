function cv = getLeftCovariance( obj )
% GETLEFTCOVARIANCE  Read left covariance operator of an 
% nlsaCovarianceOperator_gl object
%
% Modified 2014/07/17


varNames = { 'cu'  };
file = fullfile( getOperatorPath( obj ), getLeftCovarianceFile( obj ) );
load( file, varNames{ : } )
