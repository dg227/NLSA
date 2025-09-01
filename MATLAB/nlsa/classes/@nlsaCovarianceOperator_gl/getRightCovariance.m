function cv = getRightCovariance( obj )
% GETRIGHTCOVARIANCE  Read right covariance operator of an 
% nlsaCovarianceOperator_gl object
%
% Modified 2014/07/16


varNames = { 'cv'  };
file = fullfile( getOperatorPath( obj ), getRightCovarianceFile( obj ) );
load( file, varNames{ : } )
