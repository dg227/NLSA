function lambda = getCovarianceEigenvalues( obj )
% GETCOVARIANCEEIGENVALUES Get covariance eigenvalues of an nlsaModel_ssa object
%
% Modified 2016/05/31

lambda = getEigenvalues( getCovarianceOperator( obj ) );

