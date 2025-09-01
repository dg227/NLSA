function nD = getTotalDimension( obj )
% GETTOTALDIMENSION  Get the total dimension of an nlsaCovarianceOperator object
%
% Modified 2014/07/16

nD = sum( getDimension( obj ) );
