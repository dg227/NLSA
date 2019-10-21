function cv = getOperator( obj, varargin )
% GETOPERATOR  Read right covariance operator of an nlsaCovarianceOperator 
% object
%
% Modified 2014/08/05

cv = getRightCovariance( obj, varargin{ : } );
