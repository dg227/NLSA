function lambda = getEigenvalues( obj )
% GETEIGENVALUES  Get eigenvalues of an nlsaCovarianceOperator object
%
% Modified 20165/05/31

lambda = getSingularValues( obj ) .^ 2;
