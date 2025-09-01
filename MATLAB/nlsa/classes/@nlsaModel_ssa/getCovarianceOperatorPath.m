function path = getCovarianceOperatorPath( obj )
% GETCOVARIANCEOPERATORPATH  Get covariance operator path of an nlsaModel_ssa object
%
% Modified 2016/05/27

path = getPath( getCovarianceOperator( obj ) );
