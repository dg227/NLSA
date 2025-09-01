function gamma = getKoopmanEigenvalues( obj )
% GETKOOPMANEIGENVALUES Get Koopman eigenvalues of an nlsaModel object
%
% Modified 2020/06/16

gamma = getEigenvalues( getKoopmanOperator( obj ) );

