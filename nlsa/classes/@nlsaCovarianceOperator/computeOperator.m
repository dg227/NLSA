function c = computeOperator( obj, src, varargin )
% COMPUTEOPERATORC Compute right (temporal) covariance operator from 
% time-lagged embedded data src 
% 
% Modified 2014/08/05


c = computeRightCovariance( obj, src, varargin{ : } );
